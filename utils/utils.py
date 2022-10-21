import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training abstractive slides generation model."
    )
    parser.add_argument(
        "--model_name",
        help="Base model name or path to local folder containing the model.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--train_set_path",
        help="Path to the training set in json format.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--val_set_path",
        help="Path to the validation set in json format.",
        required=False,
        type=str,
    )

    parser.add_argument(
        "--test_set_path",
        help="Path to the test set in json format.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--imrad_section",
        help="The section used for model training (e.g., introduction, method, result, conclusion).",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        help="Where to store the checkpoints of the model.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--epochs",
        help="The number of training epochs.",
        required=True,
        default=10,
        type=int,
    )

    parser.add_argument(
        "--max_input_length",
        help="Maximum input length for the model",
        default=4096,
        type=int,
    )

    parser.add_argument(
        "--max_output_length",
        help="Maximum length of the generated sentence.",
        default=512,
        type=int,
    )

    parser.add_argument(
        "--batch_size",
        help="Batch size used for model training.",
        default=2,
        type=int,
    )

    parser.add_argument(
        "--dataloader_num_workers",
        help="Num workers for dataloaders.",
        default=8,
        type=int,
    )

    parser.add_argument("--use_cuda", help="Flag to enable cuda.", action="store_true")

    parser.add_argument(
        "--use_context",
        help="Flag to disable the use of context for model training.",
        action="store_true",
    )

    parser.add_argument(
        "--use_global_attention_mask",
        help="Flag to disable the use of context for model training.",
        action="store_true",
    )

    args = parser.parse_args()

    print(args)

    return args
