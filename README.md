# Score Matching Diffusion Sampler

Learns the score function (∇ log p(x)) of ETF return distributions using denoising score matching. The score at the current market state indicates the gradient of log‑density, which can be interpreted as a local momentum/mean‑reversion signal. The model is trained daily on a rolling window, and the absolute score magnitude is used to rank ETFs. Multi‑window evaluation selects the best window per ETF.

- **Score matching:** Denoising score matching with Gaussian noise
- **Network:** MLP with SiLU activations
- **Sampling:** Langevin dynamics (optional, not used for ranking)
- **Windows:** 63, 252, 504, 1008, 2016 days (best per ETF)
- **Output:** top 3 ETFs per universe by |score|

Runs daily on GitHub Actions.

## Local execution

```bash
pip install -r requirements.txt
export HF_TOKEN=<your_token>
python trainer.py
streamlit run streamlit_app.py
