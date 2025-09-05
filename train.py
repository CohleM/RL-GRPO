

import os
import re
import wandb
import random
import torch
import asyncio
import pickle
import time
import gc
import functools
import copy
import math

from collections import defaultdict
from math_verify import parse, verify
from qwen_monkey_patch import apply_qwen_patches
from dataclasses import dataclass

from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.entrypoints.engine import Engine
import sglang as l
import nest_asyncio
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from sglang.srt.patch_torch import monkey_patch_torch_reductions

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import Dataset
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed._tensor import (DTensor, Replicate, Shard,
                                       distribute_tensor)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (ColwiseParallel,
                                               PrepareModuleInput,
                                               RowwiseParallel,
                                               SequenceParallel,
                                               parallelize_module)
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_state_dict, set_model_state_dict
)
import torch.distributed.checkpoint as dcp
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp._runtime_utils import _lazy_init

def setup():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    backend = 'nccl' if device == 'cuda' else 'gloo'
    rank = int(os.environ["RANK"])
    
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if device == 'cuda':
        torch.cuda.set_device(local_rank)
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    # Initialize with explicit parameters
    dist.init_process_group(
        backend=backend, 
        world_size=world_size,
        rank=rank
    )

def get_dcp_model_state_dict(model,optimizer, data_loader, step):
    options = StateDictOptions(full_state_dict=False, cpu_offload=True) 
    return {
        "model" : get_model_state_dict(model, options=options),
        "optimizer" : optimizer.state_dict(),
        "data_loader" : data_loader.state_dict(),
        "step" : step
        }

def split_data_list(data_list, mesh):
  # we need to scatter this data_list across this mesh group, from local group 0.
    rank = mesh.get_local_rank()
    size = mesh.size()

    if rank == 0:
        data_per_ddp = math.ceil(len(data_list)/size)
    
    lists = [data_list[i * data_per_ddp: (i+1)* data_per_ddp] if rank ==0 else None for i in range(size)]
    lst = [None] # this is the output list
    dist.scatter_object_list(lst, lists, src=None, group_src=0, group=mesh.get_group())

    return lst[0]

def gather_data_list(data_list, mesh):
    # we need to scatter this data_list across this mesh group, from local group 0.
    rank = mesh.get_local_rank()
    size = mesh.size()
    lists = [None for i in range(size)] if rank==0 else None # Must be None on non-dst ranks otherwise it will call dist.gather_object in other ranks as well is None, it will be called only in the group_dst rank
    dist.gather_object(data_list,lists, group_dst=0, group=mesh.get_group()) 
    return sum(lists, []) if rank==0 else None # if not group destination, lists wil be None, won't sum

def broadcast_data_list(data_list, mesh):

    # First get the length right across the same tp group
    if mesh.get_local_rank() == 0:
        len_data_list = torch.tensor(len(data_list)).to('cuda')
    else:
        len_data_list = torch.tensor(0).to('cuda')
    
    dist.broadcast(len_data_list, group=mesh.get_group(), group_src=0)
    # then broadcast the same data_list across same tp group
    if mesh.get_local_rank() != 0:
        data_list = [None for _ in range(len_data_list)]
    
    dist.broadcast_object_list(data_list, group=mesh.get_group(), group_src=0)

    return data_list

def load_model_to_device(worker, device):
    
    if not getattr(worker.config, "offload_model", False):
        return

    _lazy_init(worker.model, worker.model)
    for handle in worker.model._all_handles:
        # print('yass')
        if handle._offload_params:
            continue
        flat_param = handle.flat_param
        handle.flat_param_to(device, non_blocking=True)
        flat_param._local_shard = flat_param.data

def load_optimizer_to_device(worker, device):

    for param_group in worker.optimizer.param_groups:
        for param in param_group["params"]:
            state = worker.optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(
                        device, non_blocking=True
                    )
                    if device=='cpu':
                        gc.collect()
                        torch.cuda.empty_cache()

# works only for llama models for now.
def prepare_llama_tp_layer(layer, device_mesh):

    parallelize_plan = {
        "self_attn" : PrepareModuleInput(
        input_kwarg_layouts = {"hidden_states" : Replicate(), "cos": Replicate(), "sin": Replicate(),"attention_mask" : Replicate() },
        desired_input_kwarg_layouts = {"hidden_states" : Replicate(), "cos": Replicate(), "sin": Replicate(), "attention_mask": Replicate()}
    ),
        "self_attn.q_proj": ColwiseParallel(use_local_output=False),
        "self_attn.k_proj": ColwiseParallel(use_local_output=False),
        "self_attn.v_proj": ColwiseParallel(use_local_output=False),
        "self_attn.o_proj": RowwiseParallel(
            # output_layouts=Shard(1)
        ),
        # "post_attention_layernorm": SequenceParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(
            # output_layouts=Shard(1)
        )
    }
    parallelize_module(
        module=layer,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_tp_model(model, mesh):
    for layer in model.model.layers:
        prepare_llama_tp_layer(layer, mesh['TP'])

    
# ----- outer block --------
    parallelize_plan = {
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(),
            # output_layouts=Shard(1)
        ),
        # "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel() # we are just specifying what it's current input layout is but internally it'll convert that Shard(1) to Replicate(), and the output will be Shard(-1)
    }
    parallelize_module(
        module=model,
        device_mesh=mesh['TP'],
        parallelize_plan=parallelize_plan
    )
    return model

def prepare_dp_model(model, mesh):

    def get_module_cls_from_name(name):
        for module in model.modules():
            if module.__class__.__name__ == name:
                return module.__class__

    transformer_layer_cls = {
        get_module_cls_from_name(name)
        for name in model._no_split_modules
    }
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_cls
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
        mixed_precision=mixed_precision,
        device_mesh=mesh['DDP', 'FSDP'],
        device_id=torch.cuda.current_device()
    )

def calc_logsumexp(tensor, mesh):

  step_size = 1024
  logsumexps = []
  for i in range(0,tensor.shape[1], step_size):
    logsumexps.append(torch.logsumexp(tensor[:,i:i+step_size,:], dim=-1))

  logsumexp = torch.cat(logsumexps, dim=-1)
  logsumexps = [torch.zeros_like(logsumexp) for _ in range(mesh['TP'].size())]

  dist.all_gather(logsumexps, logsumexp, mesh['TP'].get_group())

  logsumexps[mesh['TP'].get_local_rank()] = logsumexp # necessary to retain grad
  logsumexps = torch.stack(logsumexps, dim=-1)
  logsumexps = torch.logsumexp(logsumexps, dim=-1)
#   print(f' rank {dist.get_rank()} logsumexp requires grad ???', logsumexps.requires_grad, 'grad fn ', logsumexps.grad_fn)

  return logsumexps
 
def differentiable_all_reduce(tensor, device_mesh):

    detached_tensor = tensor.detach()
    dist.all_reduce(
        detached_tensor,
        op=dist.ReduceOp.SUM,
        group=device_mesh.get_group()
    )
    return tensor + detached_tensor - tensor.detach()

def get_output_logits(logits, actions, mesh):
  # logits must be 1,T,C actions must be 1,T

  # each process will get its own logits shard.
  # actions is the same.
  # first we need to find which action belongs to which ranks.
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  local_vocab_size = torch.LongTensor([logits.shape[-1]]).to(device)

  gathered_vocab_sizes = [torch.zeros_like(local_vocab_size) for _ in range(mesh['TP'].size())]

#   print(local_vocab_size.dtype)
  dist.all_gather(gathered_vocab_sizes, local_vocab_size, mesh['TP'].get_group())

  cu_vocab_size = torch.cumsum(
      torch.cat([torch.zeros_like(local_vocab_size)] + gathered_vocab_sizes), 0

  )
  action_device_mapping = (actions < cu_vocab_size[1:].unsqueeze(dim=-1)).to(torch.float32).argmax(dim=0) # dimension -> 1, no_of_seq
  # get rank's actions.
  # now get which sequences belong to this rank
  rank = mesh['TP'].get_local_rank()

  # get the indices of non-zero elements
  local_action_indices = torch.nonzero(action_device_mapping == rank, as_tuple=True)[0]
  # print((local_action_indices))
  local_actions = actions[:, local_action_indices] - cu_vocab_size[rank]
  # logits is B,T,C. this T dimension is shared along all the local ranks, get only the logits for loca_action_indices
  local_logits = logits[:, local_action_indices]

  action_logits = torch.zeros(actions.shape, device=torch.cuda.current_device()).type_as(local_logits)
#   print('action logits dtype', action_logits.dtype)
  action_logits[:,local_action_indices] = torch.gather(local_logits, -1, local_actions.unsqueeze(-1)).squeeze(-1)
  # now this action_logits needs to be all reduced.
  return differentiable_all_reduce(action_logits, device_mesh=mesh['TP'])

def reward_fn(output, ground_truth_ans, nums): 
    reward = 0.01
    # Get all <answer>...</answer> matches and take the last one
    answers = re.findall(r"<answer>(.*?)</answer>", output, flags=re.DOTALL)
    answer = answers[-1].strip() if answers else None

    # print('answer', answer)
    if answer:
        lhs = answer.split('=')[0]  # only get the L.H.S
        numbers = list(map(int, re.findall(r'\d+', lhs)))
        # print(lhs)
        if len(numbers) != len(nums):
            return reward
        for num in numbers:
            if num not in nums:
                return reward
        try:
            final_answer = eval(lhs, {"__builtins__": None}, {})
        except Exception:
            print('error in eval')
            return reward

        # print(f'final_answer {final_answer} ground_truth_ans {ground_truth_ans}, lhs {lhs}')
        if final_answer == ground_truth_ans:
            reward += 1.0
        else:
            reward += 0.01
    return reward

def grpo_advantage(data_list, responses_per_prompt):
  rewards = torch.FloatTensor([ex['rewards'].sum() for ex in data_list]).view(-1, responses_per_prompt)
  baseline = rewards.mean(-1)
  std = rewards.std(-1)
  advantages = (rewards - baseline.unsqueeze(-1))/ (std.unsqueeze(-1) + torch.finfo(rewards.dtype).eps)

  for ex, advantage in zip(data_list, advantages.flatten()):
    ex['advantage'] = advantage * ex['action_mask']

  return data_list


def check_mem_allocated(rank, msg):
    ans = torch.cuda.memory_allocated() / (1024**3)
    print(f'RANK {rank} MEMORY_ALLOCATED {msg} {ans}')

def calc_entropy(logits, logsumexp, mesh):
    gc.collect()
    torch.cuda.empty_cache()
    probs = torch.exp(logits - logsumexp.unsqueeze(-1))
    return logsumexp - differentiable_all_reduce((probs*logits).sum(-1), device_mesh=mesh['TP'])

def grpo_loss(minibatch, max_eps, min_eps):
    max_len = max(item['old_logprobs'].shape[0] for item in minibatch)

    def pad_tensor(t, max_len):
        return torch.nn.functional.pad(t, (max_len - t.shape[0], 0))  # pad on left

    old_logprobs = torch.stack([pad_tensor(item['old_logprobs'], max_len) for item in minibatch], dim=0)
    logprobs     = torch.stack([pad_tensor(item['current_logprobs'], max_len) for item in minibatch], dim=0)
    advantage    = torch.stack([pad_tensor(item['advantage'], max_len) for item in minibatch], dim=0).to(torch.cuda.current_device(), dtype=torch.bfloat16)
    action_mask  = torch.stack([pad_tensor(item['action_mask'], max_len) for item in minibatch], dim=0).to(torch.cuda.current_device(), dtype=torch.bfloat16)
    ratio = torch.exp(logprobs - old_logprobs)
    ob1 = ratio * advantage
    ob2 = torch.clamp(ratio, 1.0 - min_eps, 1.0 + max_eps) * advantage

    ppo_loss = -torch.min(ob1, ob2) * action_mask
    # if dist.get_rank() in [0,2]:
    #     print(f'rank ppo loss {dist.get_rank()}', ppo_loss[:, -20:])
    loss = ppo_loss.sum(dim=-1)/ action_mask.sum(dim=-1) # mean across tokens
    # entropy = (entropy_tensor.sum(dim=-1) / action_mask.sum(dim=-1))

    return loss.mean() # mean across trajectories
    
    
def avg_entropy(minibatch):
    max_len = max(item['old_logprobs'].shape[0] for item in minibatch)

    def pad_tensor(t, max_len):
        return torch.nn.functional.pad(t, (max_len - t.shape[0], 0))  # pad on left
    entropy_tensor  = torch.stack([pad_tensor(item['entropy'], max_len) for item in minibatch], dim=0).to(torch.cuda.current_device(), dtype=torch.bfloat16)
    action_mask  = torch.stack([pad_tensor(item['action_mask'], max_len) for item in minibatch], dim=0).to(torch.cuda.current_device(), dtype=torch.bfloat16)
    entropy = (entropy_tensor.sum(dim=-1) / action_mask.sum(dim=-1))

    return entropy.mean() 


class RLDataset(Dataset):
    def __init__(self, data_path, responses_per_prompt):
        self.dataset = load_dataset(data_path, split='train')
        self.responses_per_prompt = responses_per_prompt

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        # answer = ex["answer"]
        answer = ex["target"]
        nums = ex["nums"] # remove this line later

        return {
            "messages": messages,
            "answer": answer,
            "nums" : nums
        }
    def __len__(self):
        return len(self.dataset)
    
    def collate_fn(self, batch):

        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.responses_per_prompt)
        ]

class Worker:
    """
    This is the policy that we will be updating with each gradient update, we rollout using this policy's
    parameters, and we use the logprobs from this policy, we will also copy it's weights to make it old policy
    """
    
    def __init__(self, config):
        
#         self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.config = config
        device= 'cuda' if torch.cuda.is_available() else 'cpu'
        # first make a device mesh
        fsdp_size = int(int(os.environ['WORLD_SIZE']) / (config.ddp_size * config.tp_size))
        # this mesh will only be used for model partition
        self.mesh = init_device_mesh(device,(config.ddp_size,fsdp_size, config.tp_size), mesh_dim_names=["DDP", "FSDP", "TP"])
        self.dp_size = int(int(os.environ['WORLD_SIZE']) / self.config.tp_size)
        # this mesh will be used for data parallelism 
        self.device_mesh = init_device_mesh(device,(self.dp_size, config.tp_size), mesh_dim_names=["DP", "TP"])

    def prepare_optimizer(self):
        self.model.gradient_checkpointing_enable()
        if self.config.tp_size > 1:
            self.model = prepare_tp_model(self.model, self.mesh)
        
        self.model = prepare_dp_model(self.model, self.mesh)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)

        # offload the model to cpu
        load_model_to_device(self, "cpu")



class Rollout(Worker):
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.prepare_device_mesh()
        # init model using sglang
        self.prepare_env_var()

        if self.mesh["TP"].get_local_rank() == 0:
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            self.engine = Engine(
                model_path=self.config.model_name,
                dtype="bfloat16",
                tp_size=self.mesh["TP"].size(),
                mem_fraction_static=0.5,
                # enable_memory_saver=True,
                port=30000 + dist.get_rank(),
            )

        # very important to do dist.barrier() i.e block the code right here, otherwise some gpu will go on.
        dist.barrier()

    def prepare_device_mesh(self):
         
        dp_size = int(int(os.environ['WORLD_SIZE']) / self.config.tp_size)
        self.device_mesh = init_device_mesh("cuda", (dp_size, self.config.tp_size), mesh_dim_names=["DP", "TP"]) # device is on cpu cause we only need this mesh to scatter data (i.e for data parallelism)
        
    async def rollout(self, data):
        metric = defaultdict(list)

        messages,answer,nums = data['messages'], data['answer'], data['nums']
        if not self.config.apply_chat_template:
            # prompt = data['messages'] 
            # prompt = data['messages'] 

            prompt = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {data['nums']}, create an equation that equals {data['answer']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
            states = self.tokenizer.encode(prompt)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            states = self.tokenizer.encode(prompt)
        actions = [0] * len(states)
        action_mask = [0] * len(states)

        # print(ans)
        response = await self.engine.async_generate(
                prompt, sampling_params={"temperature": self.config.temperature, "max_new_tokens" : 896}
            )
        
        info = response["meta_info"]

        if not self.config.apply_chat_template:
            messages += response['text']
            # print('messages', messages)
        else:
            messages.append({'role': 'assistant', 'content': response['text']})
        
        tokenized_response = self.tokenizer.encode(response['text'])
        states.extend(tokenized_response)
        actions.extend(tokenized_response)
        action_mask.extend([1] * len(tokenized_response))
        
        # wandb metrics
        metric['response_length'] = sum(action_mask) 
        metric['trajectory_length'] = len(states)
        metric['reward'] = reward

        # sparse reward, only provide to the last token, putting extra -1 here cause later we do states[:-1]
        rewards = (len(states) -1 - 1)*[0] + [reward]
        ex = {
            'states' : torch.LongTensor(states[:-1]),
            'action_mask' : torch.LongTensor(action_mask[1:]),
            'rewards' : torch.FloatTensor(rewards),
            'actions' : torch.LongTensor(actions[1:])
        }

        return ex, messages, metric

    def __call__(self, data_list):

        if self.device_mesh['TP'].get_local_rank() ==0:
            data_list = split_data_list(data_list, mesh=self.device_mesh['DP'])

            loop = asyncio.get_event_loop()
            outputs = loop.run_until_complete(
                asyncio.gather(*(self.rollout(data) for data in data_list))
            )

            # later do this only when training
            self.engine.release_memory_occupation()
        dist.barrier()

        if self.device_mesh['TP'].get_local_rank() == 0:
            data_list, all_messages, metrics = map(list,zip(*outputs))

            # gather all the data_list 
            data_list = gather_data_list(data_list, self.device_mesh['DP'])
            # all_messages = gather_data_list(all_messages, self.device_mesh['DP'])

        if dist.get_rank() == 0:
            return data_list,all_messages, metrics
        else:
            return None, None, None


    def prepare_env_var(self):
        if (
            "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys()
        ):  # remove the use of common store for communication
            del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
        monkey_patch_torch_reductions()

        # THE reason for doing this is because, we'll store rollout worker's (sglang) weight in these TP group
        # otherwise, SGL will use all the available cuda devices.
        cuda_visible_devices = [None] * self.config.tp_size
        dist.all_gather_object(
            cuda_visible_devices, os.environ["LOCAL_RANK"], self.device_mesh["TP"].get_group()
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

    def update(self, actor):
        load_model_to_device(actor, torch.cuda.current_device())
        
        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        state_dict = get_model_state_dict(
            actor.model, options=options
        )
        # resume sglang's memory occupation
        torch.cuda.empty_cache()
        if self.device_mesh["TP"].get_local_rank() == 0:
            self.engine.resume_memory_occupation()
        
        for idx, (name, tensor) in enumerate(state_dict.items()):
            # load to gpu again, but this is a small tensor so it won't make much difference
            tensor = tensor.to(torch.cuda.current_device())
            serialized_tensor = MultiprocessingSerializer.serialize(tensor.full_tensor() if isinstance(tensor, DTensor) else tensor)
            serialized_tensors = [None] * self.device_mesh['TP'].size() if self.device_mesh['TP'].get_local_rank() == 0 else None
            
            dist.gather_object(serialized_tensor, serialized_tensors, group_dst=0, group=self.device_mesh['TP'].get_group())
            
            if self.device_mesh["TP"].get_local_rank() == 0:
                # print(serialized_tensors)
                self.engine.update_weights_from_tensor(named_tensors=[(name, LocalSerializedTensor(values=serialized_tensors))])
            
        load_model_to_device(actor, 'cpu')
        dist.barrier()



class Actor(Worker):
    def __init__(self, config):
        super().__init__(config)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, attn_implementation="eager").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name) 
        # actor will need optimizer
        self.prepare_optimizer()

    # this func will only be used for old logprobs calculation so using torch.no_grad() 
    def compute_logprobs(self, data_list, log_type):
        # let's first split the data_list again across groups
        # recplicate the data across tp dimension cause they need the same data 
        # load the model back to gpu, previously the sharded model was stored in the CPU with it's reference contained in self.model
        load_model_to_device(self, torch.cuda.current_device())
        # print('loaded model to device')
        input_ids = [item['states'] for item in data_list]
        action_input_ids = [item['actions'] for item in data_list]

        batch = {"input_ids": input_ids}
        action_batch = {"input_ids": action_input_ids}
        
        padded_input_ids = self.tokenizer.pad(batch, padding=True, padding_side='left') # make every row in the batch to have same length
        action_input_ids = self.tokenizer.pad(action_batch, padding=True, padding_side='left')['input_ids'].to('cuda')

        # print(f'rank {dist.get_rank()} and  padded input ids shape {padded_input_ids['input_ids'].shape} attention mask shape {padded_input_ids['attention_mask'].shape} ')
        padded_input_ids['input_ids'] = padded_input_ids['input_ids'].to('cuda')
        attention_mask = padded_input_ids['attention_mask'].to('cuda')
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0).to('cuda')

        logits = self.model(input_ids=padded_input_ids['input_ids'], attention_mask=attention_mask, position_ids=position_ids, use_cache=False).logits
        dist.barrier()

        B,T, vocab_size = logits.shape
        logsumexp = calc_logsumexp(logits, self.device_mesh)
        action_logits = get_output_logits(logits.view(1, B*T, vocab_size), action_input_ids.view(1,B*T), self.device_mesh).view(B,T)

        if log_type == 'current' and self.config.entropy_coeff > 0:
            check_mem_allocated(dist.get_rank(), 'before calc_entropy')
            entropy = calc_entropy(logits, logsumexp, self.device_mesh) # B,T

        logprobs = action_logits - logsumexp

        for idx in range(logprobs.shape[0]):
            # get the logprobs for only the right side of logprobs that is equals to the actions length, cause left side has been padded to match max length
            data_list[idx][f'{log_type}_logprobs'] = logprobs[idx, -len(data_list[idx]['actions']):] # now oldlogprobs and actions will have the same length as actions and action_mask
            # print( len(data_list[idx][f'{log_type}_logprobs']) == len(data_list[idx]['actions']))
            if log_type == 'current' and self.config.entropy_coeff > 0:
                data_list[idx]['entropy'] = entropy[idx, -len(data_list[idx]['actions']):]

        del logits
        gc.collect()
        torch.cuda.empty_cache() 

        return data_list     


class Trainer:
    def __init__(self, config):
        self.config = config
        check_mem_allocated(dist.get_rank(), 'before actor creation')
        self.actor = Actor(config)

        check_mem_allocated(dist.get_rank(), 'after actor creation')

        self.rollout = Rollout(config)

    def train(self):
        train_data = RLDataset(self.config.data_path, self.config.responses_per_prompt)
        train_dataloader = StatefulDataLoader(train_data, batch_size=self.config.per_rollout_size, drop_last=True, collate_fn=train_data.collate_fn)
        # construct train dataloader
        step = 0

        # init wandb
        if dist.get_rank() == 0:
            wandb.init(project=self.config.project_name, name=self.config.experiment_name, config=self.config)

        # Load the checkpoint
        if self.config.resume_steps is not None:
            # while loading models and optimizers should be on compute device 
            load_model_to_device(self.actor, torch.cuda.current_device())
            load_optimizer_to_device(self.actor, torch.cuda.current_device())
            state_dict = get_dcp_model_state_dict(self.actor.model, self.actor.optimizer, train_dataloader, step)
            dcp.load(state_dict=state_dict, checkpoint_id=f"test_folder/{self.config.resume_steps}") 

            set_model_state_dict(self.actor.model, state_dict['model']) 
            self.actor.optimizer.load_state_dict(state_dict['optimizer'])
            train_dataloader.load_state_dict(state_dict['data_loader'])
            step = state_dict['step'] 
            print(f'successfully loaded the model\n')

            load_model_to_device(self.actor, 'cpu')
            load_optimizer_to_device(self.actor, 'cpu') 
        
        for train_idx, data_list in enumerate(train_dataloader):

            trn_start = time.time() 
            if dist.get_rank() == 0:
                print(f' ----------------- TRAIN IDX {train_idx} ------------------') 

            rollout_start = time.time()
            data_list, all_messages , metrics = self.rollout(data_list) # rank 0 will only have data_list, otherwise it'll be None
            rollout_end = time.time() 

            # ------ calculate the advantage ------
            if dist.get_rank() == 0:
                data_list = grpo_advantage(data_list, self.config.responses_per_prompt)

            ## ---- old logprobs section ------
            # we've done the rollout, now let's generate the logprobs
            # since global rank 0 has the data, pass it to its other dp group members
            if self.actor.device_mesh['TP'].get_local_rank() == 0:
                data_list = split_data_list(data_list, self.actor.device_mesh['DP'])
            
            # for tp groups data must be same, so make same data
            data_list = broadcast_data_list(data_list, self.actor.device_mesh['TP'])

            with torch.no_grad():
                self.actor.model.eval()
                data_list = self.actor.compute_logprobs(data_list, log_type='old')

            ## ----- old logprobs section -----
            # divide the data_list into minibatches, each of size, i.e total_rollout_data_in_this_rank / updates_per_rollout
            mini_batch_size = len(data_list) // self.config.updates_per_rollout
            data_list = [data_list[i*mini_batch_size: (i+1)*mini_batch_size] for i in range(self.config.updates_per_rollout)]

            load_model_to_device(self.actor, torch.cuda.current_device())
            # load_optimizer_to_device(self.actor, torch.cuda.current_device()) 

            self.actor.model.train()

            dp_loss, dp_entropy,dp_grad_norm = 0.0, 0.0, 0.0
            grad_start = time.time()
            for update_step, minibatch in enumerate(data_list):
                
                # print(f' RANK {dist.get_rank()}-------- STEP: {update_step} ------------')
                minibatch = self.actor.compute_logprobs(minibatch, log_type="current")

                # we need to multiply by dp_size.
                # Explanation, in standard pre-training, where each gpu processes some part of the batch_size, the gradients are averaged automatically so that the 
                # gradients would match if they were trained on one single machine
                # Ex: if we have batch_size= 32, with 2 gpus, then if we were training on single gpu, we would do loss/total_batch_size,
                # but if we are doing it on 2 gpus, each gpu will do loss/local_batch_size (i.e 16), that's why we would do, loss/16/2 = loss/32.
                # see how averaging is only done when sequences belong to the same batch, here in this RL step they do not, so we need to cancel out the auto-averaging.
                # thus the multiplication by self.dp_size
                loss = grpo_loss(minibatch, max_eps=self.config.max_eps, min_eps=self.config.min_eps)

                if self.config.entropy_coeff > 0:
                    entropy = avg_entropy(minibatch)
                    loss = loss - self.config.entropy_coeff * entropy
                    dp_entropy += (entropy.item()/ len(data_list))

                # loss = loss 
                loss = loss * self.actor.dp_size
                print(f'RANK {dist.get_rank()}-------- STEP: {update_step} ------------ loss: {loss} len minibatch {len(minibatch)} ')
                
                dp_loss += loss.item() / len(data_list)
                # loss = loss - entropy  # add entropy

                torch.cuda.empty_cache()
                loss.backward() # when we do this, the gradients are averaged among dp groups.

                if isinstance(self.actor, FSDP):
                    grad_norm = self.actor.clip_grad_norm_(max_norm=1.0)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.model.parameters(), max_norm=1.0)
                dp_grad_norm += (grad_norm.item() / len(data_list))
                gc.collect() 
                torch.cuda.empty_cache()

                load_optimizer_to_device(self.actor, torch.cuda.current_device())
                self.actor.optimizer.step()
                self.actor.optimizer.zero_grad(set_to_none=True)
                # load_optimizer_to_device(self.actor, "cpu")


            dp_loss = gather_data_list([dp_loss], self.actor.device_mesh['DP'])
            if self.config.entropy_coeff > 0:
                dp_entropy = gather_data_list([dp_entropy], self.actor.device_mesh['DP']) 
            dp_grad_norm = gather_data_list([dp_grad_norm], self.actor.device_mesh['DP']) 

            grad_end = time.time() 
            # Log wandb metrics
            if dist.get_rank() == 0:
                # Averaging all the metrics
                dp_loss = sum(dp_loss)/len(dp_loss)
                if self.config.entropy_coeff > 0:
                    dp_entropy = sum(dp_entropy)/len(dp_entropy)
                dp_grad_norm = sum(dp_grad_norm)/len(dp_grad_norm)
                metrics = {k: sum([item[k] for item in metrics])/len(metrics) for k in metrics[0].keys()}
                metrics['loss'] = dp_loss

                if self.config.entropy_coeff > 0:
                    metrics['entropy'] = dp_entropy
                metrics['grad_norm'] = dp_grad_norm
                metrics['step'] = step
                # wandb.log(metrics)
                print('--------------- METRICS--------------- ',metrics)
                print('--------------- ROLLOUT TIME --------------- ', rollout_end - rollout_start)
                print('--------------- GRADIENT STEP TIME --------------- ', grad_end - grad_start)
                wandb.log(metrics)

            # generate rollouts. each train_batch will have length per_rollout_size x responses_per_prompt
            # first scatter the data across each ddp group
            # now let's update the slgang model with the trained model
            # takes in actor model, offload it to cpu, and piece by piece update the sglang model
            load_model_to_device(self.actor, "cpu")
            load_optimizer_to_device(self.actor, "cpu")  
            if step % self.config.checkpoint_interval == 0:

                # while checkpointing dcp expects tensors to be on compute device 
                load_model_to_device(self.actor, torch.cuda.current_device())
                load_optimizer_to_device(self.actor, torch.cuda.current_device())
                dcp.save(state_dict=get_dcp_model_state_dict(self.actor.model, self.actor.optimizer, train_dataloader, step), checkpoint_id=f"test_folder/{step}")  
                # print('successfully saved the model')

                load_model_to_device(self.actor, "cpu")
                load_optimizer_to_device(self.actor, "cpu")  
            step +=1
            self.rollout.update(self.actor)

            trn_end = time.time()
 
            if dist.get_rank() == 0:
                print('------------ ONE COMPLETE STEP TIME : ------------ ', trn_end - trn_start)
                if train_idx % 1 == 0:
                    print(all_messages[-10:])

                    with open(f'all_messages_{step}.txt', 'w') as f:
                        for item in all_messages:
                            f.write(f"--------------\n{item}\n")


def start():
    # nest_asyncio.apply()
    config = Config()
    apply_qwen_patches()
    ppo_trainer = Trainer(config)
    ppo_trainer.train()

    return

@dataclass
class Config:
    temperature: float = 1.0
    train_batch_size: int = 64
    model_name: str = 'Qwen/Qwen2.5-3B-Instruct' # make sure to add chat template to these models.
    # model_name: str = 'Qwen/Qwen2.5-3B'
    ddp_size: int = 1 
    tp_size: int = 2 
    lr: float = 1e-6
    # data_path: str = 'CohleM/olympiad_small'
    data_path: str = 'Majis699/countdown-0'
    responses_per_prompt: int = 8
    per_rollout_size: int =  8
    offload_model: bool = True
    updates_per_rollout: int = 4 
    max_eps: float = 0.2 
    min_eps: float = 0.2 
    checkpoint_interval: int = 20
    resume_steps: int = None
    project_name: str = "rl"
    experiment_name: str = "test-countdown-qwen-3b-1024-hyperbolic-new"
    entropy_coeff: float = 0.001
    # entropy_coeff: float = 0.0
    # apply_chat_template: str = True # if using False, use base model, else Instruct model
    apply_chat_template: str = False # if using False, use base model, else Instruct model
#     train_batch_size: int = 64
def main():
    # setup process groups.
    setup()
    # Initialize ppo trainer with some config
    start()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
