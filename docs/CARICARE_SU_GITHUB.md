# Repository GitHub Polysignal

La radice del repository è la cartella **Polysignal**, con `README.md`, `pyproject.toml` e `src/` direttamente al suo interno. Il repository contiene i file estratti e navigabili, non soltanto lo ZIP di consegna.

## Prima di ogni aggiornamento

1. Eseguire i controlli descritti in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
2. Verificare i file selezionati da Git: codice, documenti, report e piccoli campioni sono previsti; dataset, `.env`, sessioni, ambienti e output generati devono restare esclusi.
3. Creare un commit descrittivo e caricare sul repository `Polysignal` dell'account scelto. La visibilità iniziale per questa consegna è privata.

Il pacchetto sorgente si rigenera con `python scripts/make_release.py`. Il comando esclude i dataset per impostazione predefinita. Non usare `--include-data` per gli upload ordinari. Gli archivi privati si conservano separatamente; il manifesto storico ne documenta soltanto nomi e impronte.

La licenza MIT preesistente è conservata. Le librerie vendorizzate hanno [avvisi separati](../THIRD_PARTY_NOTICES.md). La licenza del codice non assegna automaticamente diritti sui testi social, sulla traccia o su pubblicazioni di terzi.
