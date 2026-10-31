# Polysignal — Big Data Engineering

Progetto di Francesco Simone Pirozzi e Gerardo Lauro per **Big Data Engineering 2025/2026**, Social Network Analysis Challenge, Track 2. Docente: **Vincenzo Moscato**. Report originale: giugno 2026. Raccolta preparata il 29 ottobre 2026.

PolySignal studia se i segnali social di Telegram, Reddit e X aggiungono capacità predittiva ai prezzi dei mercati Polymarket. Comprende raccolta e normalizzazione dei dati, analisi del linguaggio, associazione post–contratto, orientamento YES/NO, grafo Neo4j, valutazione e dashboard React.

## Materiali principali

- [Report tecnico PDF](pdf_latex/main.pdf), con [sorgenti LaTeX](pdf_latex/main.tex), bibliografia e figure.
- [Traccia del progetto](<Traccia_2 (1).pdf>).
- [README tecnico originale](README.md), [architettura](docs/ARCHITECTURE.md), [pipeline](docs/PIPELINE.md), [modello dati](docs/DATA_MODEL.md).
- [Corrispondenza con i requisiti](docs/SUBMISSION_CHECKLIST.md) e [scaletta della presentazione](docs/PRESENTATION.md). Le slide sono una scaletta Markdown; non è stato trovato un PowerPoint finale.
- `src/`, `scripts/`, `webapp/`, `tests/`: codice, strumenti, interfaccia e test originali.
- [Campioni e dizionario dati](data/examples/DATA_DICTIONARY.md). I dataset completi sono esclusi da questo repository.
- [Come caricare il progetto su GitHub](docs/CARICARE_SU_GITHUB.md).
- [Provenienza, contenuti ed esclusioni](docs/PROVENIENZA_PACCHETTO.md).

## Risultati documentati

Il materiale originale descrive 300 mercati risolti (100 politica, 100 finanza, 100 sport), 1.353.114 prezzi e 1.368.041 post: 1.348.965 Telegram, 16.254 Reddit, 2.822 X. I riepiloghi riportano 37.562 collegamenti accettati su 147 mercati e 35.679 stance classificate.

La conclusione riportata è che i segnali social **non danno un miglioramento predittivo consistente** rispetto al prezzo di mercato, nel dataset e nei protocolli studiati. Il corpus è fortemente concentrato su Telegram. Questi sono risultati del progetto originale, non un nuovo esperimento. Nel README e nella checklist restano anche sezioni storiche con conteggi precedenti: per la fotografia finale consultare report e `scripts/_paper_stats_*.json`.

## Avvio da una nuova macchina

Seguire l'[avvio rapido nel README](README.md) e le [istruzioni della dashboard](docs/RUN_APP.md). L'ambiente verificato è Python 3.12; il codice dichiara Python >=3.11.

Il clone non richiede né contiene il corpus completo. Senza artefatti privati la dashboard mostra un messaggio di configurazione. Per usare dati già disponibili in locale impostare `POLYSIGNAL_DATA_DIR` sulla cartella dati, oppure usare lo script di ripristino con gli archivi privati: non occorre caricarli su GitHub. I dettagli e i percorsi richiesti sono in `docs/RUN_APP.md`.

L'installazione non avvia acquisizioni, servizi Neo4j o chiamate LLM. `.env.example` è soltanto un modello: le credenziali restano nel `.env` locale ignorato da Git.

## Verifica e provenienza

Le modalità dei controlli attuali sono in [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md). Il [verbale precedente](docs/VERIFICA_PACCHETTO.md) e la [provenienza precedente](docs/PROVENIENZA_PACCHETTO.md) sono documenti storici del pacchetto ricevuto: le loro verifiche su archivi e dataset esterni non sono state ripetute in questo audit. La validazione scientifica dei risultati non rientra nel controllo di pubblicazione.
