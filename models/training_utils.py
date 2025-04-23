import torch
from torch.nn.utils import rnn

# Training utils
def build_one_instance(tokenizer, captions, motion):
    input_ids, target_ids = [], []
    bos = tokenizer.bos_token_id
    input_ids.append(bos)
    target_ids.append(-100)  # do not perform loss regression on human prompt
    texts = ''
    prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    instruction = "### Instruction:\nGenerate a motion matching the following input human motion description\n\n"
    input_text = '### Input:\n' + captions + '\n\nResponse: <Motion>'
    text = prompt + instruction + input_text
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
    # So far, the target_ids is equal to input_ids all -100
    text = '</Motion><eos>'
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    # print(one_input_id)
    # print(one_input_id)
    # print(motion)
    input_ids += motion.tolist() + one_input_id  # so the input and output should be equal, except for the human prompt.
    target_ids += motion.tolist() + one_input_id
    return input_ids, target_ids

def process_batch(tokenizer, batch_of_captions, max_tgt_len, batch_of_motions,
                  batch_of_motionscript, batch_of_audio):
    batch_input_ids, batch_target_ids = [], []
    for caption, motion, ms_segments, audio in zip(batch_of_captions, batch_of_motions,
                               batch_of_motionscript, batch_of_audio):

        # one_input_ids, one_target_ids = build_one_instance(tokenizer, caption, motion)
        one_input_ids, one_target_ids = build_training_instance_salsa(tokenizer=tokenizer,
                                                                caption=caption,
                                                                motion_tokens=motion,
                                                                motion_script_segments=ms_segments)
                                                                # audi)

        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def build_training_instance_salsa(tokenizer, caption, motion_tokens, motion_script_segments):
    """
    Build a single training instance:
    Input: coarse caption + motion script (given as a list of segments)
    Output: motion token sequence
    """

    input_ids = []
    target_ids = []
    bos_token_id = tokenizer.bos_token_id

    # Start with BOS token
    input_ids.append(bos_token_id)
    target_ids.append(-100)  # Ignore BOS during loss calculation

    # Prepare motion script (join list with <SEP>)
    joined_motion_script = format_motionscript_bins(motion_script_segments)

    # Build the full input prompt
    system_prompt = (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
    )
    instruction = "### Instruction:\nGenerate a motion matching the following human motion description and detailed motion script.\n\n"
    input_section = f"### Input:\n{caption}\n\n"
    motion_script_section = f"### MotionScript:\n{joined_motion_script}\n\n"
    response_marker = "Response: <Motion>"

    full_prompt = system_prompt + instruction + input_section + motion_script_section + response_marker

    # Tokenize the input (prompt part)
    prompt_token_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
    input_ids.extend(prompt_token_ids)
    target_ids.extend([-100] * len(prompt_token_ids))  # Mask prompt part for loss

    # Tokenize motion closing
    motion_end_token_ids = tokenizer("</Motion><eos>", add_special_tokens=False).input_ids

    # Append the motion tokens and motion end marker
    input_ids.extend(motion_tokens.tolist())
    input_ids.extend(motion_end_token_ids)

    target_ids.extend(motion_tokens.tolist()) # We already tokenized motion_tokens into LLMs tokenizer
    target_ids.extend(motion_end_token_ids)

    return input_ids, target_ids

def format_motionscript_bins(motion_script_segments, window_sec=0.5):
    """
    Prepare motion script text from list of motion descriptions.

    Args:
        motion_script_segments: List[str], each item covering `window_sec` seconds
        window_sec: float, duration each segment covers (default 0.5s)

    Returns:
        A single formatted string ready to insert into training prompt
    """
    script_lines = []
    current_time = 0.0

    for description in motion_script_segments:
        start = current_time
        end = current_time + window_sec

        if description.strip() == "":
            description_text = "<Motionless>"
        else:
            description_text = description.strip()

        line = f"{start:.1f}s-{end:.1f}s: {description_text}"
        script_lines.append(line)

        current_time = end

    # Join all segments with <SEP> separator
    joined_script = " <SEP> ".join(script_lines)
    return joined_script