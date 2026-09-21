# NeuraForge Hugging Face Demo

This directory contains the Gradio app for the NeuraForge public demo.

## Deployment Instructions

To deploy this interactive demo to Hugging Face Spaces for free:

1. Create a free account at [Hugging Face](https://huggingface.co/).
2. Create a new Space:
   - Go to https://huggingface.co/spaces and click **Create new Space**.
   - **Space name**: `NeuraForge-Demo` (or any name you prefer).
   - **License**: Choose MIT.
   - **Space SDK**: Select **Gradio**.
   - **Space hardware**: `Blank Space` or `CPU basic` (this app runs perfectly on CPU).
   - Click **Create Space**.

3. Clone the space repository locally:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/NeuraForge-Demo
   ```

4. Copy the required files into the cloned repository:
   - `app.py`
   - `requirements.txt` (append `gradio` and `pandas` to it)
   - `neuraforge/` (the entire Python package folder)
   - `results/` (the benchmark CSVs, especially `demo_model.npz` if you pre-trained one)

5. Commit and push to Hugging Face:
   ```bash
   git add .
   git commit -m "Deploy NeuraForge Demo"
   git push
   ```

Hugging Face will automatically build and launch the Gradio app!
