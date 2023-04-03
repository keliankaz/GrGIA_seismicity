import pytorch_lightning as pl
from utils import Catalog
import numpy as np
import eq
import torch


def time_torchETAS_decluster(
    event_catalog: Catalog,
):
    t_start = 0
    t_end = event_catalog.duration / np.timedelta64(1, "D")

    inter_times = np.diff(
        (
            (event_catalog.catalog.time - event_catalog.start_time)
            / np.timedelta64(1, "D")
        ).values,
        prepend=t_start,
        append=t_end,
    )

    seq = eq.data.Sequence(
        inter_times=torch.as_tensor(inter_times, dtype=torch.float32),
        t_start=t_start,
        t_nll_start=t_end
        / 5,  # 20% of the data is used to burn in the model but does not contribute to the loss
        t_end=t_end,
        mag=torch.as_tensor(event_catalog.catalog.mag.values, dtype=torch.float32),
    )
    dataset = eq.data.InMemoryDataset(sequences=[seq])
    dl = dataset.get_dataloader()

    model = eq.models.ETAS(
        mag_completeness=event_catalog.mag_completeness,
        base_rate_init=len(event_catalog) / t_end,
    )
    trainer = pl.Trainer(max_epochs=400, devices=1, accelerator="mps")
    trainer.fit(model, dl)

    mu, k, c, p, alpha = [
        getattr(model, param).item() for param in ["mu", "k", "c", "p", "alpha"]
    ]

    etas_rate = lambda t, ti, mi: mu + np.sum(
        k * 10 ** (alpha * (mi - event_catalog.mag_completeness)) / (t - ti + c) ** p
    )
    rate = np.array(
        [
            etas_rate(
                t,
                seq.arrival_times[seq.arrival_times < t].detach().numpy(),
                seq.mag[seq.arrival_times < t].detach().numpy(),
            )
            for t in seq.arrival_times.detach().numpy()
        ]
    )

    background_probability = model.mu.item() / rate

    thinned_bool = [
        True if np.random.rand() < p else False for p in background_probability
    ]
    thinned_event_catalog = event_catalog[thinned_bool]

    return Catalog(thinned_event_catalog)
