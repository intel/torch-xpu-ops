import argparse
import sys

from junitparser import JUnitXml, Error, Failure, Skipped

parser = argparse.ArgumentParser()
parser.add_argument('junitxml', nargs='+')
args = parser.parse_args()

layernorm_accuracy_failures = {
    'link': 'https://github.com/pytorch/pytorch/issues/141642',
    'cuda': 'passed',
}

# Tests were enabled for non-cuda backends by v4.49.0 (previously were
# skipped for xpu):
# https://github.com/huggingface/transformers/commit/2fa876d2d824123b80ced9d689f75a153731769b
test_cpu_offload_failures = {
    'link': 'https://github.com/huggingface/accelerate/issues/3402',
    'cuda': 'passed',
}

# That's a list of known test failures. Each listed test can have
# associated metadata in the following format:
#   failing_cases = {
#     'test_class': {
#       'test_name': {
#         'flaky': True,
#         'cuda': "passed", # or failed, or skipped
#         'link': https://github.com/org/project/issues/xxxx
#       }
#     }
#   }
# Use None if no metadata is needed.
failing_cases = {
    'tests.generation.test_logits_process.LogitsProcessorTest': {
        'test_watermarking_processor': { 'cuda': 'passed', },
    },
    'tests.generation.test_utils.GenerationIntegrationTests': {
        'test_assisted_decoding_encoder_decoder_shared_encoder': { 'cuda': 'failed', },
        'test_assisted_decoding_num_assistant_tokens_heuristic_schedule': { 'cuda': 'failed', },
        'test_assisted_generation_early_exit': { 'cuda': 'failed', },
        'test_custom_logits_processor': { 'cuda': 'failed', },
        'test_default_max_length_warning': { 'cuda': 'failed', },
        # v4.49.0+ (regression)
        # https://github.com/huggingface/transformers/commit/365fecb4d0b6c87f20b93561e11c3d4c77938012
        'test_encoder_decoder_generate_attention_mask': { 'cuda': 'failed', },
        'test_eos_token_id_int_and_list_beam_search': { 'cuda': 'failed', },
        'test_eos_token_id_int_and_list_top_k_top_sampling': { 'cuda': 'failed', },
        'test_generate_compile_fullgraph_tiny': { 'cuda': 'failed', },
        # v4.49.0+ (regression)
        # https://github.com/huggingface/transformers/commit/da334bcfa8ff7feb85138ce90ca7340e4fc6e704
        'test_generate_input_features_as_encoder_kwarg': { 'cuda': 'failed' },
        'test_max_new_tokens_encoder_decoder': { 'cuda': 'failed', },
        'test_min_length_if_input_embeds': { 'cuda': 'passed' },
        'test_model_kwarg_assisted_decoding_decoder_only': { 'cuda': 'failed' },
        'test_model_kwarg_assisted_decoding_encoder_decoder': { 'cuda': 'failed' },
        'test_model_kwarg_encoder_signature_filtering': { 'cuda': 'failed' },
        'test_prepare_inputs_for_generation_decoder_llm': { 'cuda': 'failed' },
        'test_stop_sequence_stopping_criteria': { 'cuda': 'failed' },
    },
    'tests.models.blip.test_modeling_blip.BlipTextImageModelTest': {
        'test_cpu_offload': test_cpu_offload_failures,
        'test_disk_offload_bin': test_cpu_offload_failures,
        'test_disk_offload_safetensors': test_cpu_offload_failures,
    },
    'tests.models.blip.test_modeling_blip.BlipVQAModelTest': {
        'test_cpu_offload': test_cpu_offload_failures,
        'test_disk_offload_bin': test_cpu_offload_failures,
        'test_disk_offload_safetensors': test_cpu_offload_failures,
    },
    'tests.models.dab_detr.test_modeling_dab_detr.DabDetrModelTest': {
        'test_cpu_offload': test_cpu_offload_failures,
        'test_disk_offload_bin': test_cpu_offload_failures,
        'test_disk_offload_safetensors': test_cpu_offload_failures,
    },
    'tests.models.detr.test_image_processing_detr.DetrImageProcessingTest': {
        'test_fast_is_faster_than_slow': { 'flaky': True },
    },
    'tests.models.dpt.test_modeling_dpt_auto_backbone.DPTModelTest': {
        'test_batching_equivalence': { 'flaky': True, 'cuda': 'passed' },
    },
    'tests.models.encoder_decoder.test_modeling_encoder_decoder.BartEncoderDecoderModelTest': {
        'test_save_and_load_from_pretrained': { 'flaky': True },
    },
    'tests.models.git.test_modeling_git.GitModelTest': {
        'test_generate_continue_from_past_key_values': { 'flaky': True, 'cuda': 'passed' },
    },
    'tests.models.hiera.test_modeling_hiera.HieraModelTest': {
        'test_torch_fx': layernorm_accuracy_failures,
        'test_torch_fx_output_loss': layernorm_accuracy_failures,
    },
    'tests.models.llava.test_modeling_llava.LlavaForConditionalGenerationModelTest': {
        # v4.49.0+ (regression)
        # https://github.com/huggingface/transformers/commit/bcfc9d795e1330faaa8b39ffa18732f8b40fe7c0
        'test_config': { 'cuda': 'failed' },
    },
    'tests.models.mamba.test_modeling_mamba.MambaIntegrationTests': {
        'test_simple_generate_1_cpu': { 'cuda': 'passed' },
    },
    # v4.49.0 (new test)
    # https://github.com/huggingface/transformers/commit/be2ac0916a7902e1683d708805270142257a254a
    'tests.models.paligemma.test_modeling_paligemma.PaliGemmaForConditionalGenerationModelTest': {
        'test_generate_compilation_all_outputs': { 'cuda': 'failed' },
    },
    # v4.49.0 (new test)
    # https://github.com/huggingface/transformers/commit/be2ac0916a7902e1683d708805270142257a254a
    'tests.models.paligemma2.test_modeling_paligemma2.PaliGemma2ForConditionalGenerationModelTest': {
        'test_generate_compilation_all_outputs': { 'cuda': 'failed' },
    },
    'tests.models.pix2struct.test_modeling_pix2struct.Pix2StructModelTest': {
        'test_new_cache_format_0': { 'cuda': 'passed' },
        'test_new_cache_format_1': { 'cuda': 'passed' },
        'test_new_cache_format_2': { 'cuda': 'passed' },
    },
    'tests.models.qwen2_5_vl.test_processor_qwen2_5_vl.Qwen2_5_VLProcessorTest': {
        # v4.49.0+ (new test)
        # https://github.com/huggingface/transformers/commit/15ec971b8ec999c6a511debe04ba32c115fb7413
        'test_chat_template_video_custom_sampling': { 'cuda': 'failed' },
        # v4.49.0+ (new test)
        # https://github.com/huggingface/transformers/commit/15ec971b8ec999c6a511debe04ba32c115fb7413
        'test_chat_template_video_special_processing': { 'cuda': 'failed' },
    },
    'tests.models.qwen2_vl.test_processor_qwen2_vl.Qwen2VLProcessorTest': {
        'test_chat_template_video_custom_sampling': { 'cuda': 'failed' },
        'test_chat_template_video_special_processing': { 'cuda': 'failed' },
    },
    # different failure signature than described in 'test_cpu_offload_failures'
    'tests.models.roberta.test_modeling_roberta.RobertaModelTest': {
        'test_cpu_offload': { 'cuda': 'failed' },
        'test_disk_offload_bin': { 'cuda': 'failed' },
        'test_disk_offload_safetensors': { 'cuda': 'failed' },
    },
    'tests.models.rt_detr.test_image_processing_rt_detr.RtDetrImageProcessingTest': {
        'test_fast_is_faster_than_slow': { 'flaky': True },
    },
    'tests.models.speecht5.test_modeling_speecht5.SpeechT5ForTextToSpeechIntegrationTests': {
        'test_batch_generation': { 'cuda': 'passed' },
    },
    'tests.models.vilt.test_modeling_vilt.ViltModelTest': {
        'test_cpu_offload': test_cpu_offload_failures,
        'test_disk_offload_bin': test_cpu_offload_failures,
        'test_disk_offload_safetensors': test_cpu_offload_failures,
    },
    'tests.pipelines.test_pipelines_audio_classification.AudioClassificationPipelineTests': {
        # v4.49.0+ (new test)
        # https://github.com/huggingface/transformers/commit/f19135afc77053834f1b0cdf46d9a6bf7faf7cc3
        'test_small_model_pt_fp16': { 'cuda': "failed", 'link': 'https://github.com/huggingface/transformers/issues/36340' },
    },
    'tests.pipelines.test_pipelines_automatic_speech_recognition.AutomaticSpeechRecognitionPipelineTests': {
        'test_small_model_pt_seq2seq': { 'cuda': "failed" },
    },
    'tests.pipelines.test_pipelines_common.CustomPipelineTest': {
        'test_custom_code_with_string_tokenizer': { 'cuda': "failed" },
    },
    'tests.pipelines.test_pipelines_depth_estimation.DepthEstimationPipelineTests': {
        'test_multiprocess': { 'cuda': "failed" },
    },
    'tests.pipelines.test_pipelines_image_to_text.ImageToTextPipelineTests': {
        'test_small_model_pt': { 'cuda': "failed" },
    },
    'tests.pipelines.test_pipelines_summarization.SummarizationPipelineTests': {
        'test_small_model_pt': { 'cuda': "failed" },
    },
    'tests.pipelines.test_pipelines_text_generation.TextGenerationPipelineTests': {
        # v4.49.0+ (new test)
        # https://github.com/huggingface/transformers/commit/23d782ead2fceec3e197c57de70489ccfc3bd0ee
        'test_return_dict_in_generate': { 'cuda': "failed" },
        # v4.47.0+
        'test_small_model_pt': { 'cuda': "failed" },
        # v4.49.0+ (generalized for non-cuda devices)
        # https://github.com/huggingface/transformers/commit/2fa876d2d824123b80ced9d689f75a153731769b
        'test_small_model_pt_bloom_accelerate': { 'cuda': "failed" },
        # v4.47.0+
        'test_stop_sequence_stopping_criteria': { 'cuda': "failed" },
    },
    'tests.pipelines.test_pipelines_visual_question_answering.VisualQuestionAnsweringPipelineTests': {
        'test_small_model_pt_blip2': { 'cuda': "failed" },
    },
    'tests.pipelines.test_pipelines_zero_shot_image_classification.ZeroShotImageClassificationPipelineTests': {
        'test_small_model_pt': { 'cuda': "failed" },
        'test_small_model_pt_fp16': { 'cuda': "failed" },
    },
    'tests.test_pipeline_mixin.AudioClassificationPipelineTests': {
        # v4.49.0+ (new test)
        # https://github.com/huggingface/transformers/commit/f19135afc77053834f1b0cdf46d9a6bf7faf7cc3
        'test_small_model_pt_fp16': { 'cuda': "failed", 'link': 'https://github.com/huggingface/transformers/issues/36340' },
    },
    'tests.test_pipeline_mixin.AutomaticSpeechRecognitionPipelineTests': {
        'test_small_model_pt_seq2seq': { 'cuda': "failed" },
    },
    'tests.test_pipeline_mixin.DepthEstimationPipelineTests': {
        'test_multiprocess': { 'cuda': "failed" },
    },
    'tests.test_pipeline_mixin.ImageToTextPipelineTests': {
        'test_small_model_pt': { 'cuda': "failed" },
    },
    'tests.test_pipeline_mixin.SummarizationPipelineTests': {
        'test_small_model_pt': { 'cuda': "failed" },
    },
    'tests.test_pipeline_mixin.TextGenerationPipelineTests': {
        # v4.49.0+ (new test)
        # https://github.com/huggingface/transformers/commit/23d782ead2fceec3e197c57de70489ccfc3bd0ee
        'test_return_dict_in_generate': { 'cuda': "failed" },
        # v4.47.0+
        'test_small_model_pt': { 'cuda': "failed" },
        # v4.49.0+ (generalized for non-cuda devices)
        # https://github.com/huggingface/transformers/commit/2fa876d2d824123b80ced9d689f75a153731769b
        'test_small_model_pt_bloom_accelerate': { 'cuda': "failed" },
        # v4.47.0+
        'test_stop_sequence_stopping_criteria': { 'cuda': "failed" },
    },
    'tests.test_pipeline_mixin.VisualQuestionAnsweringPipelineTests': {
        'test_small_model_pt_blip2': { 'cuda': "failed" },
    },
    'tests.test_pipeline_mixin.ZeroShotImageClassificationPipelineTests': {
        'test_small_model_pt': { 'cuda': "failed" },
        'test_small_model_pt_fp16': { 'cuda': "failed" },
    },
    'tests.trainer.test_trainer.TrainerIntegrationPrerunTest': {
        # v4.49.0+ (promoted from slow tests)
        # https://github.com/huggingface/transformers/commit/1fae54c7216e144b426e753400abdc1299d4fc74
        'test_gradient_accumulation_loss_alignment_with_model_loss': { 'cuda': "failed" },
    },
}

new_failures = []
known_failures = []
new_passes = []
flakies = []
skipped_flakies = []

def get_classname(case):
    return ' '.join(case.classname.split())

def get_name(case):
    return ' '.join(case.name.split())

def get_result(case):
    result = "passed"
    if case.result:
        if isinstance(case.result[0], Error):
            result = "error"
        elif isinstance(case.result[0], Skipped):
            result = "skipped"
        elif isinstance(case.result[0], Failure):
            result = "failed"
    return result

def get_message(case):
    if not case.result:
        return ""
    return f"{case.result[0].message.splitlines()[0]}"

def is_known_failure(classname, name):
    if classname in failing_cases and name in failing_cases[classname]:
        return True
    return False

def is_flaky(classname, name):
    if classname in failing_cases and name in failing_cases[classname]:
        _case = failing_cases[classname][name]
        if _case is None:
            return False
        return True if 'flaky' in _case and _case['flaky'] else False
    return False

def get_cuda_status(classname, name):
    if classname in failing_cases and name in failing_cases[classname]:
        _case = failing_cases[classname][name]
        if _case is None or 'cuda' not in _case:
            return ""
        return _case['cuda']
    return ""

def get_link(classname, name):
    if classname in failing_cases and name in failing_cases[classname]:
        _case = failing_cases[classname][name]
        if _case is None or 'link' not in _case:
            return ""
        link = _case['link']
        link = f"[link]({link})"
        return link
    return ""

xmls = [ JUnitXml.fromfile(f) for f in args.junitxml ]
for idx, xml in enumerate(xmls):
    for suite in xml:
        for case in suite:
            classname = get_classname(case)
            name = get_name(case)
            result = get_result(case)
            flaky = is_flaky(classname, name)
            if flaky:
                if result == "skipped":
                    skipped_flakies.append(case)
                else:
                    flakies.append(case)
            else:
                if result not in ["passed", "skipped"]:
                    if is_known_failure(classname, name):
                        known_failures.append(case)
                    else:
                        new_failures.append(case)
                else:
                    if is_known_failure(classname, name):
                        new_passes.append(case)

def print_md_row(row, print_header):
    if print_header:
        header = " | ".join([f"{key}" for key, _ in row.items()])
        print(f"| {header} |")
        header = " | ".join(["-"*len(key) for key, _ in row.items()])
        print(f"| {header} |")
    row = " | ".join([f"{value}" for _, value in row.items()])
    print(f"| {row} |")

def print_cases(cases):
    print_header = True
    for case in cases:
        classname = get_classname(case)
        name = get_name(case)
        result = get_result(case)
        message = get_message(case)
        row = {
            'Class name': classname,
            'Test name': name,
            'Status': result,
            'CUDA Status': get_cuda_status(classname, name),
            'Link': get_link(classname, name),
            'Message': message,
        }
        print_md_row(row, print_header)
        print_header = False

printed = False
def print_break(needed):
    if needed:
        print("")

if new_failures:
    print_break(printed)
    print("### New failures")
    print_cases(new_failures)
    printed = True

if known_failures:
    print_break(printed)
    print("### Known failures")
    print_cases(known_failures)
    printed = True

if new_passes:
    print_break(printed)
    print("### New passes")
    print_cases(new_passes)
    print("")
    print("> [!NOTE]")
    print("> Adjust baseline: some tests which previously failed now pass.")
    printed = True

if skipped_flakies:
    print_break(printed)
    print("### Skipped flaky tests")
    print_cases(skipped_flakies)
    print("")
    print("> [!NOTE]")
    print("> Adjust baseline: some flaky tests are now skipped.")
    printed = True

if flakies:
    print_break(printed)
    print("### Flaky tests")
    print_cases(flakies)
    printed = True

if new_failures:
    sys.exit(1)
elif new_passes:
    sys.exit(2)
elif skipped_flakies:
    sys.exit(3)

sys.exit(0)
