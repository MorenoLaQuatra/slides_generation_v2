python compute_metrics_overall.py \
    --predictions predictions/sciduet/50_128_context__sciduet__allenai__led-base-16384.json \
    --references /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    > scores/50_128_context__sciduet__allenai__led-base-16384.txt

python compute_metrics_overall.py \
    --predictions predictions/sciduet/50_128_context__sciduet__allenai__led-large-16384.json \
    --references /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    > scores/50_128_context__sciduet__allenai__led-large-16384.txt


python compute_metrics_overall.py \
    --predictions predictions/sciduet/50_128_context__sciduet__allenai__PRIMERA-arxiv.json \
    --references /data1/mlaquatra/slides/datasets/sciduet/sciduet_test.json \
    > scores/50_128_context__sciduet__allenai__PRIMERA-arxiv.txt


python compute_metrics_overall.py \
    --predictions predictions/appred/50_128_context__appred__allenai__led-base-16384.json \
    --references /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    > scores/50_128_context__appred__allenai__led-base-16384.txt

python compute_metrics_overall.py \
    --predictions predictions/appred/50_128_context__appred__allenai__led-large-16384.json \
    --references /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    > scores/50_128_context__appred__allenai__led-large-16384.txt


python compute_metrics_overall.py \
    --predictions predictions/appred/50_128_context__appred__allenai__PRIMERA-arxiv.json \
    --references /data1/mlaquatra/slides/datasets/appred/appred_test.json \
    > scores/50_128_context__appred__allenai__PRIMERA-arxiv.txt