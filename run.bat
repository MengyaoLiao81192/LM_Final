SYNTHESIZER_MODEL=Qwen/Qwen2-1.5B-Instruct \
SYNTHESIZER_BASE_URL=https://api.siliconflow.cn/v1 \
SYNTHESIZER_API_KEY=ysk-mbzpcghdipcvxmcitrkanpmiqthuveiwrgxvbawzibvbivnn \
TRAINEE_MODEL=Qwen/Qwen2-1.5B-Instruct \
TRAINEE_BASE_URL=https://api.siliconflow.cn/v1 \
TRAINEE_API_KEY=sk-mbzpcghdipcvxmcitrkanpmiqthuveiwrgxvbawzibvbivnn \
python3 -m graphgen.generate --config_file graphgen/configs/graphgen_config.yaml --output_dir cache/
