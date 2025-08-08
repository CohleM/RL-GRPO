
import os
import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from sglang.srt.entrypoints.engine import Engine
import sglang as sgl
import nest_asyncio
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from transformers import AutoModel
from sglang.srt.patch_torch import monkey_patch_torch_reductions
from transformers import AutoModelForCausalLM

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
    StateDictOptions, get_model_state_dict, get_state_dict
)

from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.model_executor.model_runner import LocalSerializedTensor


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

def prepare_llama_tp_layer(layer, device_mesh):

    parallelize_plan = {
        "input_layernorm": SequenceParallel(),
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(
            output_layouts=Shard(1)
        ),
        "post_attention_layernorm": SequenceParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(
            output_layouts=Shard(1)
        )
    }
    parallelize_module(
        module=layer,
        device_mesh=device_mesh,
        parallelize_plan=parallelize_plan
    )

def prepare_environment_variables(mesh):
    if "TORCHELASTIC_USE_AGENT_STORE" in os.environ.keys():
        del os.environ["TORCHELASTIC_USE_AGENT_STORE"]
    monkey_patch_torch_reductions()
    cuda_visible_devices = mesh["TP"].size() * [None]
    dist.all_gather_object(
        cuda_visible_devices,
        os.environ["LOCAL_RANK"],
        mesh["TP"].get_group()
    )
    # print(f' GLOBAL RNAK {dist.get_rank()} devices {cuda_visible_devices} ')
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)

def start():
    # nest_asyncio.apply()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    # torch.cuda.synchronize()
    mesh = init_device_mesh(device, (1,2,2), mesh_dim_names = ['DDP','FSDP', 'TP'])

    # print(mesh)
    # -- rollout ---
    prepare_environment_variables(mesh)
    # -- rollout ---
    
    # print(f"Global rank {dist.get_rank()}, local rank {os.environ['LOCAL_RANK']} mesh {mesh['TP']} visible devices {os.environ["CUDA_VISIBLE_DEVICES"]}\n\n  ")
    # print(torch.cuda.device_count())

    model_name = "ibm-granite/granite-3.3-2b-base"

    # ------- ROLLOUT ---------
    if mesh["TP"].get_local_rank() == 0:
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        engine = Engine(
                model_path=model_name,
                dtype="bfloat16",
                tp_size=mesh["TP"].size(),
                mem_fraction_static=0.5,
                # enable_memory_saver=True,
                port=30000 + dist.get_rank()
            )

        # print(engine)
        param_name = 'model.layers.0.self_attn.q_proj.weight'
        # if dist.get_rank() == 0:
        #     breakpoint()

        # reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        # total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        # param = engine.get_weights_by_name(param_name)[0][:5]
        # print(f"Global rank {dist.get_rank()}  param : {param} \n\n  ")
    # ------- ROLLOUT ---------

    print(f'gloabal rank {dist.get_rank()} memory allocated {torch.cuda.memory_allocated() }')
        
    # dist.barrier()

    

    # ----- rollout -----
    # Do the rollout
    if mesh["TP"].get_local_rank() == 0:
        # release memory temporarily.
        engine.release_memory_occupation()
        torch.cuda.empty_cache()
    # ----- rollout -----

    print(f'after release memory oocupation rank {dist.get_rank()} memory allocated {torch.cuda.memory_allocated() }')
    
    # load the actor model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # print(model)
    print(f'After automodel rank {dist.get_rank()} memory allocated {torch.cuda.memory_allocated() }')
    
    for layer in model.model.layers:
        prepare_llama_tp_layer(layer, mesh['TP'])

    
# ----- outer block --------
    parallelize_plan = {
        "model.embed_tokens": ColwiseParallel(
            output_layouts=Shard(1)
        ),
        "model.norm": SequenceParallel(),
        "lm_head": ColwiseParallel()
    }
    parallelize_module(
        module=model,
        device_mesh=mesh['TP'],
        parallelize_plan=parallelize_plan
    )
# ----- outer block --------

# ----------FSDP -----------
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh['DDP', 'FSDP'])
    
    fully_shard(model, mesh=mesh['DDP', 'FSDP'])

    # print(f' GLOBAL RANK {dist.get_rank()} {model.model.layers[0].self_attn.q_proj.weight.to_local().shape} memory allocated {torch.cuda.memory_allocated() / (1024 **3) }')



    # -- rollout --
    # resume memory occupation
    torch.cuda.empty_cache()
    if mesh["TP"].get_local_rank() == 0:
        engine.resume_memory_occupation()


    # -- actor changes the model---
    # let's make some changes to the first and last row
    # -- actor changes the model---
    # let's make some changes to the first and last row
    new_var = torch.tensor([1.5] * 5).to(torch.cuda.current_device())
    # print(model.model.layers[0].self_attn.q_proj.weight)
    # print(model.model.layers[0].self_attn.q_proj.weight[0][:5

    old_w = model.model.layers[0].self_attn.q_proj.weight
    local_weight = old_w.to_local().detach().clone()  
    local_weight[0][:5] = new_var

    new_mesh = old_w.device_mesh            # reuse original mesh
    placements = old_w.placements         # usually (Shard(0),)

    new_dt = DTensor.from_local(
        local_weight,
        device_mesh=new_mesh,
        placements=placements,
        run_check=False                    # safe because we kept the shape
    )

    model.model.layers[0].self_attn.q_proj.weight = torch.nn.Parameter(new_dt)

    print('after replacement', model.model.layers[0].self_attn.q_proj.weight)


    # offload to cpu
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    state_dict = get_model_state_dict(
        model, options=options
    )
    
    for idx, (name, tensor) in enumerate(state_dict.items()):
        # load to gpu again
        tensor = tensor.to(torch.cuda.current_device())
        # print(name)
        # if name == 'model.layers.0.self_attn.q_proj.weight':
            
        serialized_tensor = MultiprocessingSerializer.serialize(tensor.full_tensor() if isinstance(tensor, DTensor) else tensor)
        serialized_tensors = [None] * mesh['TP'].size() if mesh['TP'].get_local_rank() == 0 else None
        
        dist.gather_object(serialized_tensor, serialized_tensors, group_dst=0, group=mesh['TP'].get_group())
        
        if mesh["TP"].get_local_rank() == 0:
            # print(serialized_tensors)
            engine.update_weights_from_tensor(named_tensors=[(name, LocalSerializedTensor(values=serialized_tensors))])

        
        # print(f"rank {dist.get_rank()} seriliazed_tensor {serialized_tensor.shape} len_ST: {len(serialized_tensors) if isinstance(serialized_tensors,list) else serialized_tensors} ")

    dist.barrier()

    if mesh["TP"].get_local_rank() == 0:
        param_start = engine.get_weights_by_name(param_name)[0][:5]
        # param_end = engine.get_weights_by_name(param_name)[-1][:5]
        print(f"Global rank {dist.get_rank()}  param after weight update : {param_start} \n\n  ")
            
          

    return

        
def main():
    setup()
    start()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()