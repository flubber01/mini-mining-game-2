# Ollama Multi-Interface Notebook & Dev-Agent Studio

`notebook_hub.py` ist ein lokales Gradio-Studio rund um einen Ollama-Server. Es erhält die ursprünglichen Basisfunktionen (Ollama-Chat/Streaming, Notebook, Multi-File-Codegenerierung, Workspace-Import/-Export, Dateibaum, Dateiausführung und Error-Logs) und ergänzt robuste Datei-/Pfadbehandlung, Unit-Test-Generierung, Testausführung und einen validierten Self-Healing-Loop.

## Funktionen

- **Dev-Agent / Multi-File-Synthese:** Projektdateien nach Vorgabe, URL-Spezifikation und Uploads erstellen/erweitern.
- **Unit-Tests:** automatische Testdateien, `pytest` bevorzugt und `unittest` als Standardbibliothek-Fallback, strukturierte Fehlerberichte und JUnit-XML-Auswertung.
- **Zwei Reparaturschleifen:** AST-/Syntax-Self-Healing und semantisches Logic-Self-Healing anhand fehlgeschlagener Tests. Änderungen werden gesichert, erneut getestet und bei ausbleibender Verbesserung zurückgerollt. Testdateien sind standardmäßig schreibgeschützt.
- **Offline-Demo:** die Oberfläche, Datei-Pipeline, Testläufe und Self-Healing-Schleife funktionieren auch ohne Ollama. Eine Demo-Option injiziert absichtlich Logikfehler, um die Reparatur sichtbar zu machen.
- **Workspace-Explorer:** Import von Dateien, Ordnern und ZIPs (Zip-Slip-Schutz), Editor, Suche, Datei-Details, CRUD und ZIP-Export.
- **Code-Ausführung:** Live-Streaming von stdout/stderr mit Prozessgruppen-Timeout und Return-Code. Unterstützt Python sowie verfügbare Shell-/Node-/Ruby-/Perl-/PHP-Interpreter.
- **Dashboard, strukturierte Logs/Audit, Chat-Presets, Chat-Export, Notebook-Presets und CLI.**
- **Gradio 6 (aktuelles Ziel: 6.29.1+):** versionskompatible Blocks-Oberfläche; der Quellcode behält zusätzlich Fallbacks für Gradio 5.

> **Sicherheitshinweis:** Die Code-Ausführung ist kein Container oder VM-Sandbox. Programme laufen mit den Betriebssystemrechten des Hub-Prozesses. Workspace-Pfadbeschränkung, Timeout, Prozessgruppen-Kill und isolierte Umgebungsvariablen sind Schutzmaßnahmen, ersetzen aber keine Betriebssystem-Isolation. Starte keine unbekannten Projekte auf einem privilegierten Konto.

## Installation

Python 3.10+ wird empfohlen. In einer virtuellen Umgebung installieren:

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# explizit auf die aktuelle unterstützte Gradio-6-Version aktualisieren
python -m pip install --upgrade "gradio>=6.29.1,<7"
```

Ollama separat installieren/starten und ein Modell laden, zum Beispiel:

```bash
ollama serve
ollama pull qwen2.5-coder
```

Die Hub-Oberfläche startet standardmäßig auf `http://0.0.0.0:7862`. `requirements.txt` installiert Gradio aus der aktuellen 6.x-Reihe (mindestens 6.29.1; neuere 6.x-Versionen werden automatisch verwendet, Gradio 7 bleibt bis zur Validierung ausgeschlossen). Ohne erreichbaren Ollama-Server wird automatisch das integrierte Offline-Demo-Backend verwendet. `requests` ist optional; HTTP fällt auf die Python-Standardbibliothek `urllib` zurück. `pytest` ist ebenfalls optional; ohne das Paket nutzt der Test-Runner `unittest`.

```bash
python notebook_hub.py
python notebook_hub.py --mock --port 7862
```

## Schnelltest / Demo

```bash
# Eingebaute Regressionstests (ohne Ollama, temporärer Workspace)
python notebook_hub.py --selftest

# Vollständige Unit-Test-Suite für notebook_hub.py
python -m pytest -q test_notebook_hub.py
# alternativ
python -m unittest test_notebook_hub -v

# Das Demo-Projekt fehlerfrei erzeugen und testen
python notebook_hub.py --mock --synthesize "Erzeuge ein kleines Mathe- und Text-Hilfsprojekt mit Tests"

# Absichtlich fehlerhaft erzeugen und automatische Logic-Self-Healing-Runden vorführen
python notebook_hub.py --mock --inject-demo-bug --heal-rounds 3 \
  --synthesize "Erzeuge das Demo-Projekt; repariere fehlgeschlagene Tests automatisch"
```

## CLI

```text
python notebook_hub.py --help
```

Beispiele:

```bash
python notebook_hub.py --info
python notebook_hub.py --tree
python notebook_hub.py --list-models
python notebook_hub.py --run-tests --runner auto
python notebook_hub.py --run-tests --runner pytest --test-target test_mathlib.py
python notebook_hub.py --heal --heal-rounds 3
python notebook_hub.py --exec app.py --args "--verbose"
python notebook_hub.py --snippet "print(6 * 7)"
python notebook_hub.py --synthesize "Baue ein CLI-Todo-Tool mit Tests" --url http://localhost:7860 --heal
python notebook_hub.py --host 0.0.0.0 --port 7863 --auth user:password
```

## Konfiguration

Wichtige Umgebungsvariablen (CLI-Optionen können die Werte überschreiben):

| Variable | Beschreibung | Standard |
|---|---|---|
| `OLLAMA_SERVER_URL` / `OLLAMA_HOST` | Ollama-HTTP-Endpunkt | `http://localhost:11434` |
| `OMNIHACK_WORKSPACE` | Wurzelverzeichnis der Projekte | automatisch ermittelt |
| `OMNIHACK_TARGET_DIR` | Name des Zielprojekt-Ordners | `Test1` |
| `OMNIHACK_HOST` / `OMNIHACK_PORT` | Gradio-Bind-Adresse / Port | `0.0.0.0` / `7862` |
| `OMNIHACK_PYTHON` | Interpreter für Tests und Code-Ausführung | aktuelles Python |
| `OMNIHACK_MOCK_LLM=1` | Offline-Demo erzwingen | `0` |
| `OMNIHACK_AUTO_MOCK_FALLBACK=0` | automatischen Offline-Fallback abschalten | `1` |
| `OMNIHACK_LLM_TIMEOUT` | LLM-Timeout (Sekunden) | `300` |
| `OMNIHACK_EXEC_TIMEOUT` | Code-Ausführungs-Timeout (Sekunden) | `30` |
| `OMNIHACK_TEST_TIMEOUT` | Unit-Test-Timeout (Sekunden) | `120` |
| `OMNIHACK_MAX_FIX_ATTEMPTS` | maximale AST-Fix-Versuche | `3` |
| `OMNIHACK_MAX_HEAL_ROUNDS` | maximale Logic-Healing-Runden | `3` |
| `OMNIHACK_NUM_CTX` | Ollama-Kontextfenster | `8192` |

Die Dateien liegen unter `<OMNIHACK_WORKSPACE>/<OMNIHACK_TARGET_DIR>`. Fehler werden in `.system_error_log.txt`, strukturierte Aktionen in `.system_audit.jsonl` protokolliert. Überschriebene/gelöschte Dateien werden in `.backups/` gesichert. Diese internen Dateien werden in der normalen Workspace-Dateiliste ausgeblendet.

## Self-Healing-Ablauf

1. Runner sammelt Tests und Fehlerdetails (pytest inkl. JUnit-XML, sonst Konsolenparser).
2. Der Hub sucht Kandidaten über Traceback-Pfade, importierte Module, Test-/Modulnamen und Fehlerausgabe.
3. Das Modell bekommt Quelldateien sowie die zugehörigen **read-only** Testdateien und soll komplette Quell-Dateien zurückgeben.
4. Pfade werden innerhalb von `Test1` validiert; Testdateien sind standardmäßig gesperrt; Python-Änderungen müssen den AST-Check bestehen.
5. Bestehende Dateien werden gesichert und die Tests neu ausgeführt.
6. Eine Reparatur wird nur übernommen, wenn der Fehler-Score sinkt und nicht weniger Tests gesammelt wurden; andernfalls werden Änderungen zurückgerollt.

Der optionale Haken „Testdatei-Änderungen erlauben“ schaltet den Anti-Cheat-Schutz bewusst aus; die Option bleibt standardmäßig deaktiviert.
