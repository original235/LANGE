import os
import argparse
from data_postprocess import DataPostprocess

def parse_args():
    parser = argparse.ArgumentParser(description='Data processing pipeline')
    
    # Environment variables
    parser.add_argument('--cuda_devices', type=str, default='0',
                        help='CUDA device ID')
    parser.add_argument('--worker_method', type=str, default='spawn',
                        help='VLLM worker multiprocessing method')
    
    # Pipeline parameters
    parser.add_argument('--num_gpu', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95,
                        help='GPU memory utilization ratio')
    parser.add_argument('--threshold', type=float, default=7.0,
                        help='Threshold for data generation')
    parser.add_argument('--ablation', type=str, default=None,
                        help='Ablation type')
    
    # Paths
    parser.add_argument('--model_path', type=str, default='',
                        help='Path to the model (required for vllm mode)')
    parser.add_argument('--seed_path', type=str, required=True,
                        help='Path to seed data')
    parser.add_argument('--new_instances_path', type=str, required=True,
                        help='Path for new instances')
    parser.add_argument('--filtered_path', type=str, required=True,
                        help='Path for filtered instances')
    parser.add_argument('--visual_path', type=str, required=False,
                        help='Path for visualization')
    parser.add_argument('--review_path', type=str, required=True,
                        help='Path for review output')

    # LLM inference mode
    parser.add_argument('--mode', type=str, default='vllm', choices=['vllm', 'api'],
                        help='mode for LLM inference (vllm or api)')

    # LLM config
    parser.add_argument('--apikey', type=str, default='',
                        help='apikey for LLM config (required for api mode)')
    parser.add_argument('--url', type=str, default='https://coding.dashscope.aliyuncs.com/v1',
                        help='url for LLM config (required for api mode)')
    parser.add_argument('--modelname', type=str, default='qwen3.5-plus',
                        help='modelname for LLM config (required for api mode)')
    parser.add_argument('--api_num_worker', type=int, default=4,
                        help='Number of concurrent API workers for api mode')

    args = parser.parse_args()
    
    if args.mode == 'vllm' and not args.model_path:
        parser.error("--model_path is required when --mode is 'vllm'")
    if args.mode == 'api':
        if not args.apikey or not args.url or not args.modelname:
            parser.error("--apikey, --url, and --modelname are required when --mode is 'api'")
            
    return args

if __name__ == '__main__':
    args = parse_args()
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = args.worker_method
    
    if args.mode == 'api':
        from review_instances_api import DataReviewer
        from generate_instances_api import DataSource
    else:
        from review_instances_vllm import DataReviewer
        from generate_instances_vllm import DataSource
    from filter_data import DataFilter

    if args.mode == 'api':
        gener = DataSource(
            model_name=args.modelname,
            api_key=args.apikey,
            base_url=args.url,
            data_path=args.seed_path,
            output_path=args.new_instances_path,
            threshold=args.threshold,
            api_num_worker=args.api_num_worker
        )
    else:
        gener = DataSource(
            model_path=args.model_path,
            data_path=args.seed_path,
            output_path=args.new_instances_path,
            num_gpu=args.num_gpu,
            gpu_memory_utilization=args.gpu_memory_utilization,
            threshold=args.threshold
        )
    gener(ablation=args.ablation)

    datafilter = DataFilter(
        data_path=args.new_instances_path,
        output_path=args.filtered_path,
    )
    datafilter()

    if args.mode == 'api':
        reviewer = DataReviewer(
            model_name=args.modelname,
            api_key=args.apikey,
            base_url=args.url,
            data_path=args.filtered_path,
            output_path=args.review_path,
            seed=False,
            api_num_worker=args.api_num_worker
        )
    else:
        reviewer = DataReviewer(
            model_path=args.model_path,
            data_path=args.filtered_path,
            output_path=args.review_path,
            num_gpu=args.num_gpu,
            gpu_memory_utilization=args.gpu_memory_utilization,
            seed=False
        )
    reviewer()

    data_postprocess = DataPostprocess(
        data_path=args.review_path,
        ablation=None,
        threshold=args.threshold
    )
    data_postprocess()