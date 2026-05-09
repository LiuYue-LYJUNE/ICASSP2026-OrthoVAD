import torch


def test(test_loader, model, device, args):
    result = {}

    model.eval() if hasattr(model, 'eval') else None

    for _, data in enumerate(test_loader):
        feature, data_video_name = data
        feature = feature.to(device)
        seq_len = torch.sum(torch.max(torch.abs(feature), dim=2)[0] > 0, 1)

        with torch.no_grad():
            outputs = model(feature, seq_len)
            if isinstance(outputs, tuple):
                element_logits = outputs[0]
            else:
                element_logits = outputs

            element_logits = torch.mean(element_logits, dim=0)

        element_logits = element_logits.cpu().data.numpy().reshape(-1)
        result[data_video_name[0]] = element_logits

    return result
