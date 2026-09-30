"""Encode Beauty titles/descriptions; tensor row i always represents item i."""

import argparse
import html
import json
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_PLM_DIR = DATA_DIR.parents[2] / 'LLM'


def clean_text(value):
    if isinstance(value, list):
        return ' '.join(filter(None, (clean_text(part) for part in value)))
    if value is None:
        return ''
    text = html.unescape(str(value))
    text = re.sub(r'<[^>]+>', ' ', text)
    return ' '.join(text.split())


def get_item_text(item_path, features):
    with item_path.open(encoding='utf-8') as stream:
        items = json.load(stream)
    expected_ids = {str(i) for i in range(len(items))}
    if not items or set(items) != expected_ids:
        raise ValueError('Item IDs must be consecutive strings from 0 to N-1')
    texts = []
    fallback_count = 0
    for item_id in range(len(items)):
        item = items[str(item_id)]
        parts = [clean_text(item.get(feature)) for feature in features]
        text = ' '.join(part.rstrip('.') + '.' for part in parts if part)
        if not text:
            text = f"Product {item.get('asin') or item_id}."
            fallback_count += 1
        texts.append(text)
    print(f'Items: {len(texts)}; items using ASIN fallback: {fallback_count}')
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--item_path', type=Path, default=DATA_DIR / 'Beauty.item.json')
    parser.add_argument('--plm_dir', type=Path, default=DEFAULT_PLM_DIR)
    parser.add_argument('--plm_name', default='sentence-t5-base', choices=['sentence-t5-base'])
    parser.add_argument('--device', default=None, help='e.g. cuda:0 or cpu; auto-detect by default')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_sent_len', type=int, default=512)
    parser.add_argument('--features', nargs='+', default=['title', 'description'])
    parser.add_argument('--output_path', type=Path, default=None)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_sent_len <= 0:
        parser.error('batch_size and max_sent_len must be positive')

    model_path = args.plm_dir / args.plm_name
    if not model_path.is_dir():
        parser.error(f'Model directory not found: {model_path}')
    texts = get_item_text(args.item_path, args.features)

    import torch
    from sentence_transformers import SentenceTransformer

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SentenceTransformer(str(model_path.resolve()), device=device)
    model.max_seq_length = args.max_sent_len
    model.eval()
    # Preserve Sentence-T5's pooling, projection and normalization modules.
    with torch.no_grad():
        embeddings = model.encode(texts, batch_size=args.batch_size,
                                  show_progress_bar=True, convert_to_tensor=True)
    embeddings = embeddings.detach().cpu().float()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(texts):
        raise ValueError(f'Unexpected embedding shape: {tuple(embeddings.shape)}')
    if not torch.isfinite(embeddings).all():
        raise ValueError('Embeddings contain NaN or infinity')
    output_path = args.output_path or DATA_DIR / f'text_embed_{args.plm_name}.pt'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)
    print(f'Saved {tuple(embeddings.shape)} float32 tensor to {output_path.resolve()}')


if __name__ == '__main__':
    main()
