> **Documento storico ricevuto nello ZIP del progetto.** Le attività e i dati qui descritti appartengono alla raccolta precedente. L’audit GitHub corrente parte dal solo `BDE-PolySignal-GitHub.zip`, senza gli archivi completi; non conferma ricerche su altri dischi/account né ripete la verifica dei dataset. Per i controlli attuali vedere [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

# Provenienza e copertura — 29 ottobre 2026

## Fonti raccolte

- Cartella locale originale `Progetto_BDE`: codice Python, dashboard React con librerie incluse, 86 file di test/fixture, documentazione, configurazioni modello, traccia PDF, report PDF e LaTeX, dati e risultati.
- Materiali archiviati di sviluppo in `_cleanup_trash`: raccolti separatamente nello ZIP storico, inclusi gli appunti Big Data. Il loro stato storico è preservato.
- Dossier locale `Polymarket_tesi_Orlando/2026-09-15`: panoramica, catalogo fonti, verifica delle fonti e nove paper PDF. Conservato come materiale di preparazione della tesi, con data originale; non prova assegnazione formale o approvazione delle proposte.
- Ricerca per nomi pertinenti in Documents e Downloads: nessun altro materiale del progetto individuato nella copertura effettuata.
- Google Drive collegato e Gmail principale: ricerche mirate per PolySignal, BDE, Big Data Engineering e nomi della traccia senza risultati pertinenti. Non è una verifica di Teams, WhatsApp o della posta universitaria.

## Organizzazione

- `PolySignal/`: repository con codice e documentazione, traccia, report e campioni.
- `01-dati-dashboard.zip`: risultati analitici, mercati e dati necessari alla dashboard.
- `02-dataset-completo.zip`: intero export normale e intero export partizionato presenti nell'originale.
- `03-dati-pipeline.zip`: corpus normalizzato, collegamenti, corpus di linking ed evidenze accettate/scartate rimaste fuori dal primo ZIP, inclusa la grande tabella CSV.
- `04-materiali-storici.zip`: vecchi script, prototipo, prompt di sviluppo, appunti e note storiche.
- `05-dati-grezzi.zip`: cartella canonica `data/raw`, salvo eventuali file sensibili indicati nel registro.
- `Tesi-e-paper/`: dossier e paper del settembre 2026.

## Esclusioni deliberate

Sono esclusi credenziali `.env`, impostazioni private, sessioni Telegram/browser, ambiente `.venv`, strumenti scaricati `.tools`, cache, log, backup e vecchi archivi di consegna. Gli originali non sono stati modificati.

La cartella temporanea **`data/staging`** contiene 1.647.267 file, circa 7,62 GB, fra acquisizioni di lavoro, copie e stati intermedi: non fa parte del pacchetto. Sono invece inclusi integralmente i dataset finali e le tabelle canoniche utili alla riproduzione dell'analisi. La cache e il backup pre-integrazione non vengono replicati. Questa è una raccolta per repository e consegna, non un'immagine integrale del disco.

`BDE_Homework` riguarda MovieLens 25M ed è un lavoro distinto; la vecchia cartella `Tesi_FrancescoSimonePirozzi` riguarda altri materiali. Non sono confluiti in PolySignal.

## Interventi sulla copia

Conservati il codice applicativo e la pipeline. Aggiunti guida italiana, istruzioni GitHub, manifesti SHA-256 e script di ripristino dei dati. Aggiornati `.gitignore` e `.gitattributes`; aggiunto un collegamento iniziale nel README originale. Ripuliti i campi di autenticazione nel modello `.env.example`. Resi portabili i collegamenti locali nel dossier tesi. Le fonti restano inalterate.

L'inventario dettagliato e gli hash sono nella radice del pacchetto, insieme all'elenco delle esclusioni. Non sono stati avviati nuovi scraping, inferenze a pagamento o messaggi a terzi.
