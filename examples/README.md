# Examples Directory

This directory contains reference images and instructions for testing model inference.

## Providing Test Images

You can place test waste images in this directory to evaluate model predictions using the CLI:

```bash
python -m src.main predict --image examples/sample_cardboard.jpg --top-k 3
```

### Supported Image Formats
- JPEG / JPG (`.jpg`, `.jpeg`)
- Portable Network Graphics (`.png`)

### Requirements for Best Inference Accuracy
1. The waste item should be centered and reasonably illuminated.
2. The background should ideally be plain or consistent with typical recycling collection conditions.
3. Supported waste categories:
   - `cardboard`
   - `glass`
   - `metal`
   - `paper`
   - `plastic`
   - `trash`
