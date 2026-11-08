from pathlib import Path
import numpy as np

def sample_posterior(
    posterior,
    x,
    theta_true=None,
    n_samples=5000,
    label="Held-out",
    save_path=None
):
    ntest = len(x)

    median_all = np.zeros((ntest, 2))
    lo_all = np.zeros((ntest, 2))
    hi_all = np.zeros((ntest, 2))
    all_samples = np.empty((ntest, n_samples, 2), dtype=np.float32)

    # 1. Sample posterior and calculate summaries
    for j in range(ntest):

        samples = posterior.sample(
            (n_samples,),
            x=x[j:j+1],
            show_progress_bars=False
        ).detach().cpu().numpy()

        median_all[j] = np.median(samples, axis=0)
        lo_all[j] = np.percentile(samples, 16, axis=0)
        hi_all[j] = np.percentile(samples, 84, axis=0)
        all_samples[j] = samples

        if (j + 1) % 50 == 0:
            print(f"{label}: {j + 1}/{ntest} inferred")

    results = {
        "median": median_all,
        "lo": lo_all,
        "hi": hi_all
    }

    # 2. Include true parameters
    if theta_true is not None:
        results["theta_true"] = (
            theta_true.detach().cpu().numpy()
            if hasattr(theta_true, "detach")
            else np.asarray(theta_true)
        )

    # 3. Save results
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(save_path, **results)
        np.savez(
            save_path.parent / "posterior_samples.npz",
            samples=all_samples
        )
        print("Saved:", save_path)

    return results
