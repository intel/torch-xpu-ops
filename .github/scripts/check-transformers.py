import argparse
import sys

from junitparser import JUnitXml, Error, Failure, Skipped

parser = argparse.ArgumentParser()
parser.add_argument('junitxml', nargs='+')
args = parser.parse_args()

failing_cases = {
    'tests.benchmark.test_benchmark.BenchmarkTest': {
        'test_inference_encoder_decoder_with_configs': None,
        'test_inference_fp16': None,
        'test_inference_no_configs': None,
        'test_inference_no_configs_only_pretrain': None,
        'test_inference_no_model_no_architectures': None,
        'test_inference_torchscript': None,
        'test_inference_with_configs': None,
        'test_save_csv_files': None,
        'test_trace_memory': None,
        'test_train_encoder_decoder_with_configs': None,
        'test_train_no_configs': None,
        'test_train_no_configs_fp16': None,
        'test_train_with_configs': None,
    },
    'tests.generation.test_logits_process.LogitsProcessorTest': {
        'test_watermarking_processor': None,
    },
    'tests.generation.test_utils.GenerationIntegrationTests': {
        'test_assisted_decoding_encoder_decoder_shared_encoder': None,
        'test_assisted_decoding_num_assistant_tokens_heuristic_schedule': None,
        'test_assisted_generation_early_exit': None,
        'test_custom_logits_processor': None,
        'test_default_max_length_warning': None,
        'test_eos_token_id_int_and_list_beam_search': None,
        'test_eos_token_id_int_and_list_top_k_top_sampling': None,
        'test_generate_compile_fullgraph_tiny': None,
        'test_generated_length_assisted_generation': None,
        'test_max_new_tokens_encoder_decoder': None,
        'test_min_length_if_input_embeds': None,
        'test_model_kwarg_assisted_decoding_decoder_only': None,
        'test_model_kwarg_assisted_decoding_encoder_decoder': None,
        'test_model_kwarg_encoder_signature_filtering': None,
        'test_prepare_inputs_for_generation_decoder_llm': None,
        'test_stop_sequence_stopping_criteria': None,
    },
    'tests.models.detr.test_image_processing_detr.DetrImageProcessingTest': {
        'test_fast_is_faster_than_slow': { 'flaky': True },
    },
    'tests.models.dpt.test_modeling_dpt_auto_backbone.DPTModelTest': {
        'test_batching_equivalence': { 'flaky': True },
    },
    'tests.models.fuyu.test_modeling_fuyu.FuyuModelTest': {
        'test_prompt_lookup_decoding_matches_greedy_search': { 'flaky': True },
    },
    'tests.models.git.test_modeling_git.GitModelTest': {
        'test_generate_continue_from_past_key_values': { 'flaky': True },
        'test_inputs_embeds_matches_input_ids': None,
    },
    'tests.models.hiera.test_modeling_hiera.HieraModelTest': {
        'test_torch_fx': None,
        'test_torch_fx_output_loss': None,
    },
    'tests.models.mamba.test_modeling_mamba.MambaIntegrationTests': {
        'test_simple_generate_1_cpu': None,
    },
    'tests.models.pix2struct.test_modeling_pix2struct.Pix2StructModelTest': {
        'test_new_cache_format_0': None,
        'test_new_cache_format_1': None,
        'test_new_cache_format_2': None,
    },
    'tests.models.speecht5.test_modeling_speecht5.SpeechT5ForTextToSpeechIntegrationTests': {
        'test_batch_generation': None,
    },
    'tests.pipelines.test_pipelines_automatic_speech_recognition.AutomaticSpeechRecognitionPipelineTests': {
        'test_small_model_pt_seq2seq': None,
    },
    'tests.pipelines.test_pipelines_common.CustomPipelineTest': {
        'test_custom_code_with_string_tokenizer': None,
    },
    'tests.pipelines.test_pipelines_depth_estimation.DepthEstimationPipelineTests': {
        'test_multiprocess': None,
    },
    'tests.pipelines.test_pipelines_image_to_text.ImageToTextPipelineTests': {
        'test_small_model_pt': None,
    },
    'tests.pipelines.test_pipelines_summarization.SummarizationPipelineTests': {
        'test_small_model_pt': None,
    },
    'tests.pipelines.test_pipelines_text_generation.TextGenerationPipelineTests': {
        'test_small_model_pt': None,
        'test_stop_sequence_stopping_criteria': None,
    },
    'tests.pipelines.test_pipelines_video_classification.VideoClassificationPipelineTests': {
        'test_small_model_pt': None,
    },
    'tests.pipelines.test_pipelines_visual_question_answering.VisualQuestionAnsweringPipelineTests': {
        'test_small_model_pt_blip2': None,
    },
    'tests.pipelines.test_pipelines_zero_shot_image_classification.ZeroShotImageClassificationPipelineTests': {
        'test_small_model_pt': None,
        'test_small_model_pt_fp16': None,
    },
    'tests.test_pipeline_mixin.AutomaticSpeechRecognitionPipelineTests': {
        'test_small_model_pt_seq2seq': None,
    },
    'tests.test_pipeline_mixin.DepthEstimationPipelineTests': {
        'test_multiprocess': None,
    },
    'tests.test_pipeline_mixin.ImageToTextPipelineTests': {
        'test_small_model_pt': None,
    },
    'tests.test_pipeline_mixin.SummarizationPipelineTests': {
        'test_small_model_pt': None,
    },
    'tests.test_pipeline_mixin.TextGenerationPipelineTests': {
        'test_small_model_pt': None,
        'test_stop_sequence_stopping_criteria': None,
    },
    'tests.test_pipeline_mixin.VideoClassificationPipelineTests': {
        'test_small_model_pt': None,
    },
    'tests.test_pipeline_mixin.VisualQuestionAnsweringPipelineTests': {
        'test_small_model_pt_blip2': None,
    },
    'tests.test_pipeline_mixin.ZeroShotImageClassificationPipelineTests': {
        'test_small_model_pt': None,
        'test_small_model_pt_fp16': None,
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
