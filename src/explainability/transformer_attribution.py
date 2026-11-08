def token_occlusion_attributions(model, tokenizer, text, target_index=None):
    try:
        import torch
    except ImportError as error:
        raise ImportError("Install PyTorch to use transformer token occlusion") from error
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    with torch.no_grad():
        original_logits = model(**encoded).logits[0]
        if target_index is None:
            target_index = int(original_logits.argmax().item())
        original_score = float(torch.softmax(original_logits, dim=-1)[target_index].item())

    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    unknown_id = next((
        token_id for token_id in (tokenizer.unk_token_id, tokenizer.mask_token_id, tokenizer.pad_token_id)
        if token_id is not None
    ), None)
    if unknown_id is None:
        raise ValueError("Tokenizer must provide an unknown, mask, or pad token for occlusion")

    attributions = []
    for token_index, (start, end) in enumerate(offsets):
        if start == end:
            continue
        occluded_ids = input_ids.clone()
        occluded_ids[0, token_index] = unknown_id
        model_inputs = {"input_ids": occluded_ids}
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        with torch.no_grad():
            occluded_logits = model(**model_inputs).logits[0]
            occluded_score = float(torch.softmax(occluded_logits, dim=-1)[target_index].item())
        attributions.append({
            "start": start,
            "end": end,
            "text": text[start:end],
            "score": original_score - occluded_score,
        })
    return {
        "method": "TOKEN_OCCLUSION",
        "target_index": target_index,
        "target_score": original_score,
        "attributions": attributions,
        "perturbation": "replace one token with tokenizer unknown/mask/pad token",
    }


def integrated_gradients_attributions(model, tokenizer, text, target_index=None, steps=32):
    try:
        import torch
        from captum.attr import LayerIntegratedGradients
    except ImportError as error:
        raise ImportError("Install Captum and PyTorch to use Integrated Gradients") from error
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if target_index is None:
        with torch.no_grad():
            target_index = int(model(**encoded).logits[0].argmax().item())

    def forward_embeds(inputs_embeds):
        return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits

    layer = model.get_input_embeddings()
    baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id or 0)
    attributor = LayerIntegratedGradients(forward_embeds, layer)
    attributes, delta = attributor.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_index,
        additional_forward_args=(),
        n_steps=steps,
        return_convergence_delta=True,
    )
    token_scores = attributes.sum(dim=-1).squeeze(0).detach().cpu().tolist()
    attributions = [
        {"start": start, "end": end, "text": text[start:end], "score": float(token_scores[index])}
        for index, (start, end) in enumerate(offsets)
        if start < end
    ]
    return {
        "method": "INTEGRATED_GRADIENTS",
        "target_index": target_index,
        "attributions": attributions,
        "convergence_delta": float(delta.detach().cpu().reshape(-1)[0].item()),
        "baseline": "padding token IDs; assess linguistic validity before interpreting",
    }
