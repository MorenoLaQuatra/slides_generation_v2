# ------------------------------------------
# Models with context
# ------------------------------------------

# SciDuet

python test.py \
    --trained_model_suffix _sciduet_allenai__led-base-16384 \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_context__sciduet__allenai__led-base-16384.json \
    --use_global_attention_mask \
    --use_context \
    --use_cuda

python test.py \
    --trained_model_suffix _sciduet_allenai__led-large-16384 \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_context__sciduet__allenai__led-large-16384.json \
    --use_global_attention_mask \
    --use_context \
    --use_cuda

python test.py \
    --trained_model_suffix _sciduet_allenai__PRIMERA-arxiv \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_context__sciduet__allenai__PRIMERA-arxiv.json \
    --use_global_attention_mask \
    --use_context \
    --use_cuda


# APPreD

python test.py \
    --trained_model_suffix _appred_allenai__led-base-16384 \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_context__appred__allenai__led-base-16384.json \
    --use_global_attention_mask \
    --use_context \
    --use_cuda

python test.py \
    --trained_model_suffix _appred_allenai__led-large-16384 \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_context__appred__allenai__led-large-16384.json \
    --use_global_attention_mask \
    --use_context \
    --use_cuda

python test.py \
    --trained_model_suffix _appred_allenai__PRIMERA-arxiv \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_context__appred__allenai__PRIMERA-arxiv.json \
    --use_global_attention_mask \
    --use_context \
    --use_cuda

# SciDuet

python test.py \
    --trained_model_suffix _sciduet_facebook__bart-large \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_context__sciduet__facebook__bart-large.json \
    --max_input_length 1024 \
    --use_context \
    --use_cuda

python test.py \
    --trained_model_suffix _sciduet_google__pegasus-large \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_context__sciduet__google__pegasus-large.json \
    --max_input_length 1024 \
    --use_context \
    --use_cuda

# APPreD

python test.py \
    --trained_model_suffix _appred_facebook__bart-large \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_context__appred__facebook__bart-large.json \
    --max_input_length 1024 \
    --use_context \
    --use_cuda

python test.py \
    --trained_model_suffix _appred_google__pegasus-large \
    --trained_models_root models/50_128_context/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_context__appred__google__pegasus-large.json \
    --max_input_length 1024 \
    --use_context \
    --use_cuda

:'
# ------------------------------------------
# Models NO context
# ------------------------------------------

# SciDuet

python test.py \
    --trained_model_suffix _sciduet_allenai__led-base-16384 \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_nocontext__sciduet__allenai__led-base-16384.json \
    --use_global_attention_mask \
    --use_cuda

python test.py \
    --trained_model_suffix _sciduet_allenai__led-large-16384 \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_nocontext__sciduet__allenai__led-large-16384.json \
    --use_global_attention_mask \
    --use_cuda

python test.py \
    --trained_model_suffix _sciduet_allenai__PRIMERA-arxiv \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_nocontext__sciduet__allenai__PRIMERA-arxiv.json \
    --use_global_attention_mask \
    --use_cuda


# APPreD

python test.py \
    --trained_model_suffix _appred_allenai__led-base-16384 \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_nocontext__appred__allenai__led-base-16384.json \
    --use_global_attention_mask \
    --use_cuda

python test.py \
    --trained_model_suffix _appred_allenai__led-large-16384 \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_nocontext__appred__allenai__led-large-16384.json \
    --use_global_attention_mask \
    --use_cuda

python test.py \
    --trained_model_suffix _appred_allenai__PRIMERA-arxiv \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_nocontext__appred__allenai__PRIMERA-arxiv.json \
    --use_global_attention_mask \
    --use_cuda


# ------------------------------------------
# BASELINES
# ------------------------------------------

# SciDuet

python test.py \
    --trained_model_suffix _sciduet_facebook__bart-large \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_nocontext__sciduet__facebook__bart-large.json \
    --max_input_length 1024 \
    --use_cuda

python test.py \
    --trained_model_suffix _sciduet_google__pegasus-large \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --output_file_path predictions/sciduet/50_128_nocontext__sciduet__google__pegasus-large.json \
    --max_input_length 1024 \
    --use_cuda

# APPreD

python test.py \
    --trained_model_suffix _appred_facebook__bart-large \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_nocontext__appred__facebook__bart-large.json \
    --max_input_length 1024 \
    --use_cuda

python test.py \
    --trained_model_suffix _appred_google__pegasus-large \
    --trained_models_root models/50_128_nocontext/ \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --output_file_path predictions/appred/50_128_nocontext__appred__google__pegasus-large.json \
    --max_input_length 1024 \
    --use_cuda
'

