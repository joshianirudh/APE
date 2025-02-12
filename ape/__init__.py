def enable_attention_prefill_prefix(model_name, model):
    if "llama" in model_name:
        from .ape_llama import enable_llama_attention_prefill_prefix
        enable_llama_attention_prefill_prefix(model)
    elif "mistral" in model_name:
        from .ape_mistral import enable_mistral_attention_prefill_prefix
        enable_mistral_attention_prefill_prefix(model)
    elif "gemma" in model_name:
        from .ape_gemma import enable_gemma_attention_prefill_prefix
        enable_gemma_attention_prefill_prefix(model)

def enable_attention_prefill_context(model_name, model):
    if "llama" in model_name:
        from .ape_llama import enable_llama_attention_prefill_context
        enable_llama_attention_prefill_context(model)
    elif "mistral" in model_name:
        from .ape_mistral import enable_mistral_attention_prefill_context
        enable_mistral_attention_prefill_context(model)
    elif "gemma" in model_name:
        from .ape_gemma import enable_gemma_attention_prefill_context
        enable_gemma_attention_prefill_context(model)

def enable_attention_prefill_query(model_name, model, temperature, scale):
    if "llama" in model_name:
        from .ape_llama import enable_llama_attention_prefill_query
        enable_llama_attention_prefill_query(model, temperature, scale)
    elif "mistral" in model_name:
        from .ape_mistral import enable_mistral_attention_prefill_query
        enable_mistral_attention_prefill_query(model, temperature, scale)
    elif "gemma" in model_name:
        from .ape_gemma import enable_gemma_attention_prefill_query
        enable_gemma_attention_prefill_query(model, temperature, scale)