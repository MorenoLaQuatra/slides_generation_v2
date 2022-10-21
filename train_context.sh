# -------------------------------------------------------
# LED-base
# -------------------------------------------------------
python train.py --model_name allenai/led-base-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_sciduet_allenai__led-base-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-base-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_sciduet_allenai__led-base-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-base-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_sciduet_allenai__led-base-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-base-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_sciduet_allenai__led-base-16384/ \
    --use_context \
    --epochs 10

# -------------------------------------------------------
# APPreD
# -------------------------------------------------------


python train.py --model_name allenai/led-base-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_appred_allenai__led-base-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-base-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_appred_allenai__led-base-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-base-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_appred_allenai__led-base-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-base-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_appred_allenai__led-base-16384/ \
    --use_context \
    --epochs 10

# -------------------------------------------------------
# LED-large
# -------------------------------------------------------

python train.py --model_name allenai/led-large-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_sciduet_allenai__led-large-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-large-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_sciduet_allenai__led-large-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-large-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_sciduet_allenai__led-large-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-large-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_sciduet_allenai__led-large-16384/ \
    --use_context \
    --epochs 10

# -------------------------------------------------------
# APPreD
# -------------------------------------------------------


python train.py --model_name allenai/led-large-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_appred_allenai__led-large-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-large-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_appred_allenai__led-large-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-large-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_appred_allenai__led-large-16384/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/led-large-16384 \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_appred_allenai__led-large-16384/ \
    --use_context \
    --epochs 10



# -------------------------------------------------------
# PRIMERA
# -------------------------------------------------------

python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_sciduet_allenai__PRIMERA-arxiv/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_sciduet_allenai__PRIMERA-arxiv/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_sciduet_allenai__PRIMERA-arxiv/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_sciduet_allenai__PRIMERA-arxiv/ \
    --use_context \
    --epochs 10

# -------------------------------------------------------
# APPreD
# -------------------------------------------------------


python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_appred_allenai__PRIMERA-arxiv/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_appred_allenai__PRIMERA-arxiv/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_appred_allenai__PRIMERA-arxiv/ \
    --use_context \
    --epochs 10

python train.py --model_name allenai/PRIMERA-arxiv \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_appred_allenai__PRIMERA-arxiv/ \
    --use_context \
    --epochs 10


# -------------------------------------------------------
# PEGASUS
# -------------------------------------------------------

python train.py --model_name google/pegasus-large \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_sciduet_google__pegasus-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name google/pegasus-large \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_sciduet_google__pegasus-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name google/pegasus-large \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_sciduet_google__pegasus-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name google/pegasus-large \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_sciduet_google__pegasus-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

# -------------------------------------------------------
# APPreD
# -------------------------------------------------------


python train.py --model_name google/pegasus-large \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_appred_google__pegasus-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name google/pegasus-large \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_appred_google__pegasus-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name google/pegasus-large \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_appred_google__pegasus-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name google/pegasus-large \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_appred_google__pegasus-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024



# -------------------------------------------------------
# BART
# -------------------------------------------------------

python train.py --model_name facebook/bart-large \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_sciduet_facebook__bart-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name facebook/bart-large \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_sciduet_facebook__bart-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name facebook/bart-large \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_sciduet_facebook__bart-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name facebook/bart-large \
    --train_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_sciduet_facebook__bart-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

# -------------------------------------------------------
# APPreD
# -------------------------------------------------------


python train.py --model_name facebook/bart-large \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section introduction \
    --output_dir models/50_128_context/introduction_appred_facebook__bart-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name facebook/bart-large \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section method \
    --output_dir models/50_128_context/method_appred_facebook__bart-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name facebook/bart-large \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section result \
    --output_dir models/50_128_context/result_appred_facebook__bart-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024

python train.py --model_name facebook/bart-large \
    --train_set_path /data1/mlaquatra/slides/datasets/appred/appred_train.json \
    --val_set_path /data1/mlaquatra/slides/datasets/appred/appred_eval.json \
    --test_set_path /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    --imrad_section conclusion \
    --output_dir models/50_128_context/conclusion_appred_facebook__bart-large/ \
    --use_context \
    --epochs 10 \
    --max_input_length 1024