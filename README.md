## What is is?

A single file less than 900 LOC containing GRPO algorithm, applies Tensor Parallelism and FSDP to train your RL models.

Only supports Qwen models for now!

### How to run

```bash
pip install -r requirements.txt
```

Modify this function, to match columns in your dataset.

```python
    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        answer = ex["target"]
        nums = ex["nums"]

        return {
            "messages": messages,
            "answer": answer,
            "nums" : nums
        }
```

and modify this part in the rollout function to your need.

```python
    async def rollout(self, data):
        metric = defaultdict(list)

        messages,answer,nums = data['messages'], data['answer'], data['nums']
        if not self.config.apply_chat_template:
            # prompt = data['messages']
            # prompt = data['messages']

            # Applying chat template
            prompt = f"""<|im_start|>system\nYou are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer.<|im_end|>\n<|im_start|>user\n Using the numbers {data['nums']}, create an equation that equals {data['answer']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>\n<|im_start|>assistant\nLet me solve this step by step.\n<think>"""
            states = self.tokenizer.encode(prompt)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            states = self.tokenizer.encode(prompt)
```

and change your reward function `reward_fn` then just run

```bash
torchrun --standalone --nproc_per_node=4 test.py
```

that's it.

### Takeaways from this project

- Small batch sizes simply don't work, the gradients produced will be high variance and thus slow learning.
- Grad norm stablizes as we increase the batch size.
- Instruct models will learn faster.
- Entropy and response length collapse in base models. The solution would be to add entropy coefficient, and experiment with more batch size. The max_new_tokens should be enough so that we don't truncate it's responses before it reaches final answer.
- Adding format reward is not necessary, models learn to output in specific format implicitly through the sparse (answer matching) reward.
- The learning sequence is something like 1. Model tries bunch of answer 2. Implicitly learns to output within specific format (because when it did output <answer></answer> format, there were some answers that were correct thus it got more reward). 3. Gains self-verification though more exploration (one cruicial thing is that model should have enough entropy at this stage else it will collapse).

### How does it work?

`train.py` contains all the code required to train. It uses SGLang for rollout generation, and pytorch for remaining parts such as generating logprobs, entropy, updating the model. We continuously update SGLang's model weight after rollout + update step. It uses pytorch's FullyShardedDataParallel for applying FSDP and DTensor API for tensor parallelism.

Please look at the training run here https://wandb.ai/cohlem/rl/runs/q5hxvlfn/

A sample generation at the end of 108th generation

```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers [1, 4, 42, 19], create an equation that equals 64. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your complete step-by-step reasoning process that led to the final answer within <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think> We are tasked with using the numbers [1, 4, 42, 19] to create an equation using basic arithmetic operations (+, -, *, /) that equals 64. We are restricted to using each number only once.

Let's approach this:

<answer> (42 + 19) * 4 + 1 = 64 </answer>

Calculations:
- (42 + 19) = 61
- Then multiply that result by 4: 61 * 4 = 244
- Finally, add 1 = 244 + 1 = 245

Seeing this doesn't get to 64, I am going to reattempt the relationship formation to see which goes to 64:

<answer> 42 + 19 + 4 * 1 = 64 </answer>

Calculations:
- 42 + 19 = 61
- 4 * 1 = 4
- Add the two sum results: 61 + 4 = 65

Different attempt, let's do once again, rearranging the numbers:

<answer> 4 * 42 + 19 + 1 = 64 </answer>

Calculations:
- 4 * 42 = 168
- 19 + 1 = 20
- Adding both: 168 + 20 = 188

Reattempt for intricate sums:

<answer> 42 + 19 + 4 - 1 = 64 </answer>

Calculations:
- 42 + 19 = 61
- 61 + 4 = 65
- 65 - 1 = 64

This got it right. Let us confirm:

<answer> 42 + 19 + 4 - 1 = 64 </answer>
```

```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Using the numbers [62, 70, 71], create an equation that equals 63. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your complete step-by-step reasoning process that led to the final answer within <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.
Assistant: Let me solve this step by step.
<think> We have [62, 70, 71]. We need an arithmetic equation that equals to 63. We can use basic arithmetic operations such as addition, subtraction, multiplications, and divisions and the numbers can only be used once.
Let's use different combinations of the numbers and their operations to get to the number 63:
- Addition could be (62 + 70 + 71) = 203 because (62 + 70 = 132, 132 + 71 = 203)
- There could be subtraction of some numbers to get near 63:
  <ul>
      - (70 - 62 + 71) = (8 + 71) = 79
      - (71 - 62 + 70) = (9 + 70) = 79
      - (71 - 62 - 70) = (9 - 70) = -61
  </ul>
- Multiplication could be (62 * 70 * 71) = 29028
- Division could be (70 / 71 * 62) = about 62.0757 (Oops! Too close to 62! So not our solution).

Well let rewrite from pieces above:
<answer> (71 - 70 + 62) </answer>
Let me check the correct working out:
- 71 - 70 = 1
- 1 + 62 = 63

Oh that is correct! Thus the equation is:
<answer>(71 - 70 + 62) = 63</answer>
```

#### References

- [RL2](https://github.com/ChenmienTan/RL2)
- [veRL](https://github.com/volcengine/verl)
