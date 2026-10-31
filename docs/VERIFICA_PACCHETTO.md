> **Documento storico ricevuto nello ZIP del progetto.** Le attività e i dati qui descritti appartengono alla raccolta precedente. L’audit GitHub corrente parte dal solo `BDE-PolySignal-GitHub.zip`, senza gli archivi completi; non conferma ricerche su altri dischi/account né ripete la verifica dei dataset. Per i controlli attuali vedere [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# Verifica del pacchetto — 29 ottobre 2026

- **27 test superati**, selezionati fra test del pacchetto, configurazione, CLI, percorsi dati e validità JSON del server. Eseguiti su una copia isolata, senza `.env` originale e senza test di rete.
- Ripristino riuscito dello ZIP della dashboard mediante lo script incluso; verifica SHA-256 prima dell'estrazione e per ogni file estratto.
- Dashboard ricostruita dai file ripristinati: **300 mercati**, **335 schede di evidenza**. Controllo `_validate` superato e JSON serializzabile senza NaN/Infinity.
- Il contatore `postsLoaded` restituito dalla dashboard è **1,357,296**. Il report storico descrive 1.368.041 post: si conservano entrambi i valori con la rispettiva provenienza, senza considerarli intercambiabili.
- **2720 voci nei cinque ZIP** confrontate con l'hash del file di origine; lettura completa degli archivi e integrità CRC controllata.
- **3049 corrispondenze fonte/destinazione** registrate in `INVENTARIO_FONTI.csv` (comprendono i campioni copiati anche nel repository).
- Controllata la selezione Git su una copia: codice e campioni inclusi, dataset ripristinati ignorati; nessun file del repository oltre 25 MiB. Il maggiore è la traccia PDF, 5.661.607 byte.
- Controlli per nomi di file sensibili e formati noti di chiavi nei testi esportati. Credenziali locali e sessioni escluse; nessun segreto riportato nei registri.

Questa verifica riguarda confezionamento, ripristino e costruzione dei dati della dashboard. Non è una nuova validazione scientifica dei risultati, né una nuova acquisizione social, né una prova su una macchina priva di Python. Per i test è stato riutilizzato l'interprete dell'ambiente originale, con codice e dati presi dalla copia isolata.
