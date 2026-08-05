#!/usr/bin/env python3
"""
SRA Agent Node — Intent-Routing Command Parser (Phase 2, Step 3)

Replaces the single-intent command_parser.py with a full intent-routing agent:
  - Classifies: delivery | query | status | stop | unknown
  - Injects live DB context into every LLM call (single-pass, no second LLM call)
  - Always publishes a spoken response to /sra/tts/speak
  - delivery intent: also publishes structured command to /sra/ai/command
  - Gates on /sra/system/state — only accepts commands when state is IDLE
    (fail-open if state is None to prevent startup deadlocks)

Subscribes:
  /sra/operator/raw_command  (String)
  /sra/system/state          (String JSON)

Publishes:
  /sra/ai/command            (String JSON)  — delivery intents only
  /sra/tts/speak             (String)       — every response
  /sra/alerts/events         (String JSON)  — errors and warnings
"""

import json
import os
from datetime import datetime, timezone

import psycopg2
import requests
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# Configuration — all values from environment variables (.env)
# ---------------------------------------------------------------------------
OLLAMA_URL   = os.getenv("SRA_OLLAMA_URL",   "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("SRA_OLLAMA_MODEL", "llama3.1:8b-instruct-q4_K_M")
MAX_QUANTITY = int(os.getenv("SRA_MAX_DELIVERY_QUANTITY", "6"))

DB_CONFIG = {
    "host":     os.getenv("SRA_DB_HOST",     "localhost"),
    "dbname":   os.getenv("SRA_DB_NAME",     "sra_db"),
    "user":     os.getenv("SRA_DB_USER",     "sra_user"),
    "password": os.getenv("SRA_DB_PASSWORD", ""),
    "port":     int(os.getenv("SRA_DB_PORT", "5432")),
}

VALID_BACK_COLORS  = {"blue", "red", "black"}
VALID_FRONT_COLORS = {"blue", "red"}
VALID_FUSE_POS     = {"none", "upper", "lower", "both"}
VALID_STATIONS     = {"station_A", "station_B"}

# ---------------------------------------------------------------------------
# System prompt — multi-intent agent.
#
# Context injection: the literal string "{context_block}" is replaced at
# runtime by _call_llm() using str.replace() — NOT str.format() — so that
# the JSON examples below (which contain curly braces) are never misread
# as Python format placeholders.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an intent-routing agent for an industrial robot assistant (SRA)
that manages fuse box delivery in an Industry 4.0 production line.

Each fuse box has a configuration code:  back_color-front_color-fuse_position
  back_color:    blue | red | black
  front_color:   blue | red
  fuse_position: none | upper | lower | both
Valid destinations: estacion A = station_A, estacion B = station_B

━━━ LIVE DB CONTEXT (injected before this call) ━━━
{context_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ INTENT TYPES ━━━

1. delivery — operator wants to send boxes to a station
   Required fields: config_code, quantity, destination
   Example: "Envíame dos cajas azul-rojo-ambos a la estación A"

2. query — operator asks about inventory or stock levels
   Use the [DB_CONTEXT] provided to answer accurately.
   Example: "¿Cuántas cajas azul-rojo-ambos tenemos?"

3. status — operator asks about system state or recent activity
   Use the [DB_CONTEXT] provided to answer.
   Example: "¿Cómo está el sistema?" / "¿Hay tareas activas?"

4. stop — operator wants to halt all operations immediately
   Keywords: STOP, PARADA, DETENER, EMERGENCIA, ALTO
   Example: "¡Parada de emergencia!"

5. unknown — message is not a valid command for this system
   Example: "¿Qué hora es?" / unintelligible input

━━━ STRICT RULES ━━━
- Use the DB CONTEXT above to answer query and status intents — do NOT guess.
- For delivery: config_code + quantity + destination are ALL required.
- Do NOT invent values not explicitly stated by the operator.
- Respond ONLY with a valid JSON object. No explanation, markdown, or extra text.
- The "response" field is spoken aloud — make it natural, concise, and correct.
   Match the operator's language (Spanish → "es").
- For delivery: response must confirm what was understood (config, quantity, destination).
- For query/status: response must directly answer using the DB_CONTEXT data.
- For unknown: response must politely ask the operator to clarify.
- For stop: response must confirm the stop command was received.

━━━ FUSE POSITION DISAMBIGUATION ━━━
"dos"/"two" almost always = NUMBER OF BOXES, not fuse position.
Spanish fuse position mappings:
  none  → "sin fusibles", "ninguno", "sin fusible"
  upper → "fusible superior", "arriba", "solo arriba"
  lower → "fusible inferior", "abajo", "solo abajo"
  both  → "ambos fusibles", "los dos fusibles", "arriba y abajo",
           "superior e inferior", "dos fusibles", "con fusibles", "doble fusible"

━━━ OUTPUT FORMAT ━━━
Respond ONLY with this JSON — no other text:
{
  "intent":      "delivery | query | status | stop | unknown",
  "config_code": "back-front-fuse" or null,
  "quantity":    <integer> or null,
  "destination": "station_A | station_B" or null,
  "confidence":  <float 0.0 to 1.0>,
  "response":    "<spoken response for the operator>",
  "language":    "es",
  "error":       null or "<short reason if command is invalid>"
}

━━━ EXAMPLES ━━━
Input: "Envíame dos cajas azul-rojo-ambos a la estación A"
Output: {"intent":"delivery","config_code":"blue-red-both","quantity":2,"destination":"station_A","confidence":0.98,"response":"Entendido. Enviando 2 cajas azul-rojo-ambos a la estación A.","language":"es","error":null}

Input: "¿Cuántas cajas azul-rojo-ambos tenemos?"
Output: {"intent":"query","config_code":"blue-red-both","quantity":null,"destination":null,"confidence":1.0,"response":"Tenemos 7 cajas azul-rojo-ambos disponibles actualmente.","language":"es","error":null}

Input: "Cuantas cajas hemos producido hoy?"
Output: {"intent":"query","config_code":null,"quantity":null,"destination":null,"confidence":1.0,"response":"Hemos producido 42 cajas en total de todas las configuraciones.","language":"es","error":null}

Input: "¿Cuál es el estado del sistema?"
Output: {"intent":"status","config_code":null,"quantity":null,"destination":null,"confidence":1.0,"response":"El sistema está en estado IDLE, listo para recibir comandos. No hay tareas activas.","language":"es","error":null}

Input: "Para todo"
Output: {"intent":"stop","config_code":null,"quantity":null,"destination":null,"confidence":0.95,"response":"Deteniendo operación. El sistema regresará a IDLE.","language":"es","error":null}

Input: "bla bla noise test"
Output: {"intent":"unknown","config_code":null,"quantity":null,"destination":null,"confidence":0.1,"response":"No entendí el comando. Por favor, indique una entrega, consulta o estado.","language":"es","error":"Input is not a valid delivery command or query."}
"""


class CommandParserNode(Node):

    def __init__(self):
        super().__init__("sra_command_parser")

        self.model        = OLLAMA_MODEL
        self.url          = OLLAMA_URL
        self.max_quantity = MAX_QUANTITY

        # Local mirror of system state — updated by /sra/system/state subscription.
        # None means state machine has not published yet (fail-open).
        self.current_state: str | None = None

        # Persistent DB connection — reused across all context queries.
        self._db_conn = None
        self._connect_db()

        # ── Subscriptions ────────────────────────────────────────────────
        self.create_subscription(
            String, "/sra/system/state",        self._state_callback,   10
        )
        self.create_subscription(
            String, "/sra/operator/raw_command", self._command_callback, 10
        )

        # ── Publishers ───────────────────────────────────────────────────
        self.pub_command = self.create_publisher(String, "/sra/ai/command",    10)
        self.pub_tts     = self.create_publisher(String, "/sra/tts/speak",     10)
        self.pub_alert   = self.create_publisher(String, "/sra/alerts/events", 10)
        self.parse_result_pub = self.create_publisher(String, "/sra/ai/parse_result", 10)

        self.get_logger().info(
            f"Agent ready. Model: {self.model}  MaxQty: {self.max_quantity}"
        )

    # -----------------------------------------------------------------------
    # Subscription callbacks
    # -----------------------------------------------------------------------

    def _state_callback(self, msg: String) -> None:
        """Keep a local mirror of system state for behavioral gating."""
        try:
            data = json.loads(msg.data)
            self.current_state = data.get("state")
        except json.JSONDecodeError:
            self.get_logger().warn("Could not parse /sra/system/state message.")

    def _command_callback(self, msg: String) -> None:
        """
        Main entry point:
          1. Behavioral gate — reject if not IDLE (fail-open if state unknown).
          2. Build live DB context block.
          3. Call LLM with injected context.
          4. Route result by intent.
        """

        # ── 1. Behavioral gate ────────────────────────────────────────────
        if self.current_state is not None and self.current_state != "IDLE":
            self.get_logger().warn(
                f'Ignored command (state={self.current_state}): '
                f'"{msg.data[:60]}"'
            )
            return

        if self.current_state is None:
            self.get_logger().warn(
                "State unknown — processing command (fail-open)."
            )

        raw_text = msg.data.strip()
        self.get_logger().info(f'Received: "{raw_text}"')

        # ── 2. Build context → call LLM ──────────────────────────────────
        context_block = self._build_context_block()
        result        = self._call_llm(raw_text, context_block)

        intent        = result.get("intent",   "unknown")
        response_text = result.get("response", "")
        error         = result.get("error")

        # Fallback spoken response if LLM returned no response field
        if not response_text:
            response_text = (
                f"Error del sistema: {error}" if error
                else "No pude procesar el comando."
            )

        # ── 3. Always speak the response ─────────────────────────────────
        self._publish_tts(response_text)

        # ── 4. Route by intent ────────────────────────────────────────────
        if intent == "delivery" and not error:
            validated = self._validate_delivery(result)
            if validated.get("error"):
                err_msg = validated["error"]
                self.get_logger().warn(f"Delivery validation failed: {err_msg}")
                self._publish_tts(f"Comando de entrega inválido: {err_msg}")
                self._publish_alert("warning", err_msg)
            else:
                out = String()
                out.data = json.dumps(validated, ensure_ascii=False)
                self.pub_command.publish(out)
                self.get_logger().info(
                    f"Delivery → {validated['config_code']} "
                    f"×{validated['quantity']} → {validated['destination']}"
                )

        elif intent == "stop":
            self.get_logger().warn("STOP command received from operator.")
            self._publish_alert("critical", "Operator requested STOP.")

        elif intent in ("query", "status"):
            self.get_logger().info(f"Intent={intent} — response sent to TTS.")

        else:
            # unknown intent or top-level LLM error
            self.get_logger().warn(f"Unknown/error: {error or 'no error detail'}")
            if error:
                self._publish_alert("warning", error)
                
        # Always notify the state machine that parsing is complete,
        # regardless of intent. This is the only exit from PARSING state.
        result_msg = String()
        result_msg.data = json.dumps({
            "intent":     intent,
            "error":      result.get("error"),
            "confidence": result.get("confidence", 0.0),
        })
        self.parse_result_pub.publish(result_msg)

    # -----------------------------------------------------------------------
    # DB context injection
    # -----------------------------------------------------------------------

    def _build_context_block(self) -> str:
        """
        Query the DB once and return a compact, LLM-readable context string.

        Example output:
          INVENTORY (available = stock minus active reservations):
            blue-black-both:7 | blue-black-lower:3 | blue-red-both:5 | ...
          TOTAL AVAILABLE: 42 boxes across all configurations
          ACTIVE TASKS: 1  |  QUEUED: 0
          LAST DELIVERY: blue-red-both × 2 → station_A (5 min ago)

        Notes:
        - Available = physical stock minus active inventory_reservations.
        - After the read-only query, rollback() closes the implicit transaction
          (required because autocommit=False).
        - If the DB is unavailable, returns a safe fallback string so the LLM
          can still classify the intent (delivery validation will catch errors).
        """
        # Lazy reconnect if connection was lost
        if self._db_conn is None:
            self._connect_db()
        if self._db_conn is None:
            return "[DB CONTEXT: unavailable — connection failed]"

        try:
            with self._db_conn.cursor() as cur:

                # Available inventory per configuration
                cur.execute("""
                    SELECT
                        c.config_code,
                        i.quantity - COALESCE(r.reserved, 0) AS available
                    FROM configurations c
                    JOIN inventory i ON c.id = i.config_id
                    LEFT JOIN (
                        SELECT config_id, SUM(quantity) AS reserved
                        FROM inventory_reservations
                        GROUP BY config_id
                    ) r ON c.id = r.config_id
                    ORDER BY c.config_code
                """)
                rows = cur.fetchall()
                inv_parts      = [f"{code}:{avail}" for code, avail in rows]
                inv_line       = " | ".join(inv_parts)
                total_available = sum(avail for _, avail in rows)

                # Active and queued task counts
                cur.execute("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'in_progress') AS active,
                        COUNT(*) FILTER (WHERE status = 'queued')      AS queued
                    FROM tasks
                    WHERE status IN ('in_progress', 'queued')
                """)
                active_count, queued_count = cur.fetchone()

                # Last completed delivery
                cur.execute("""
                    SELECT config_code, quantity, destination, completed_at
                    FROM tasks
                    WHERE status = 'completed'
                    ORDER BY completed_at DESC
                    LIMIT 1
                """)
                last = cur.fetchone()
                if last:
                    l_code, l_qty, l_dest, l_time = last
                    now = datetime.now(timezone.utc)
                    if l_time.tzinfo is None:
                        l_time = l_time.replace(tzinfo=timezone.utc)
                    delta_min = int((now - l_time).total_seconds() / 60)
                    last_str = f"{l_code} × {l_qty} → {l_dest} ({delta_min} min ago)"
                else:
                    last_str = "none yet"

            # Close the implicit read-only transaction (autocommit=False)
            self._db_conn.rollback()

            return (
                f"INVENTORY (available = stock minus active reservations):\n"
                f"  {inv_line}\n"
                f"TOTAL AVAILABLE: {total_available} boxes across all configurations\n"
                f"ACTIVE TASKS: {active_count}  |  QUEUED: {queued_count}\n"
                f"LAST DELIVERY: {last_str}"
            )

        except Exception as exc:
            self.get_logger().warn(f"DB context query failed: {exc}")
            try:
                self._db_conn.rollback()
            except Exception:
                pass
            self._db_conn = None  # Trigger reconnect on next call
            return "[DB CONTEXT: query failed — LLM will proceed without live data]"

    # -----------------------------------------------------------------------
    # LLM interaction
    # -----------------------------------------------------------------------

    def _call_llm(self, text: str, context_block: str) -> dict:
        """
        Inject context into the system prompt, send to Ollama, parse JSON.

        Context injection uses str.replace() — NOT str.format() — so that
        the JSON examples in SYSTEM_PROMPT (which contain curly braces)
        are never misinterpreted as Python format placeholders.

        Never raises — always returns a dict (error_result on failure).
        """
        prompt = SYSTEM_PROMPT.replace("{context_block}", context_block)

        try:
            response = requests.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user",   "content": text},
                    ],
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.json()["message"]["content"].strip()

            ## Strip markdown code fences if the LLM wraps its response
            #if content.startswith("```"):
            #    content = content.split("```")[1]
            #    if content.startswith("json"):
            #        content = content[4:]
            #content = content.strip()
            
            # Extract the JSON object by structure, not by a specific preamble style —
            # handles code fences, bare headers, and any other wrapping text uniformly.
            start = content.find("{")
            end = content.rfind("}")
            if start == -1 or end == -1 or end < start:
                raise json.JSONDecodeError("No JSON object found in LLM output", content, 0)
            content = content[start:end + 1]
            
            print("============== DEBUG: ==============")
            print(content) #DEBUG =============================================================================================
            return json.loads(content)

        except requests.exceptions.Timeout:
            return self._error_result("Ollama request timed out")
        except json.JSONDecodeError as exc:
            return self._error_result(f"LLM returned invalid JSON: {exc}")
        except Exception as exc:
            return self._error_result(f"Unexpected error: {exc}")

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_delivery(self, parsed: dict) -> dict:
        """
        Validate delivery-specific fields after the LLM confirms intent=delivery.
        Returns a new dict with the error field set if validation fails.
        Does not mutate the input dict.
        """
        code = parsed.get("config_code", "")
        qty  = parsed.get("quantity",    0)
        dest = parsed.get("destination", "")

        parts = str(code).split("-")
        if len(parts) != 3:
            return {**parsed, "error": f"Invalid config_code format: '{code}'"}

        back, front, fuse = parts
        if back  not in VALID_BACK_COLORS:
            return {**parsed, "error": f"Unknown back color: '{back}'"}
        if front not in VALID_FRONT_COLORS:
            return {**parsed, "error": f"Unknown front color: '{front}'"}
        if fuse  not in VALID_FUSE_POS:
            return {**parsed, "error": f"Unknown fuse position: '{fuse}'"}
        if dest  not in VALID_STATIONS:
            return {**parsed, "error": f"Unknown destination: '{dest}'"}

        if not isinstance(qty, int) or qty < 1 or qty > self.max_quantity:
            return {**parsed, "error": (
                f"Invalid quantity: {qty} "
                f"(must be integer between 1 and {self.max_quantity})"
            )}

        # Confidence is diagnostic only — clamp to valid range, never block
        conf = parsed.get("confidence", 0.0)
        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
            return {**parsed, "confidence": 0.0}

        return parsed

    # -----------------------------------------------------------------------
    # DB connection management
    # -----------------------------------------------------------------------

    def _connect_db(self) -> None:
        """
        Open a persistent psycopg2 connection.
        autocommit=False — all reads use explicit rollback() to close
        implicit transactions; writes use explicit commit()/rollback().
        Logs but never raises on failure.
        """
        try:
            self._db_conn = psycopg2.connect(**DB_CONFIG)
            self._db_conn.autocommit = False
            self.get_logger().info("DB connection established.")
        except Exception as exc:
            self._db_conn = None
            self.get_logger().warn(
                f"DB connection failed: {exc} — context injection disabled."
            )

    # -----------------------------------------------------------------------
    # Publisher helpers
    # -----------------------------------------------------------------------

    def _publish_tts(self, text: str) -> None:
        """Publish plain text to /sra/tts/speak (tts_node expects a raw string)."""
        if not text:
            return
        msg      = String()
        msg.data = text
        self.pub_tts.publish(msg)

    def _publish_alert(self, level: str, message: str) -> None:
        """Publish a structured JSON alert to /sra/alerts/events."""
        alert = {
            "level":     level,
            "message":   message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        msg      = String()
        msg.data = json.dumps(alert)
        self.pub_alert.publish(msg)

    # -----------------------------------------------------------------------
    # Static helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _error_result(reason: str) -> dict:
        """
        Standard error dict returned when the LLM call itself fails.
        Includes a spoken response so _publish_tts always has something to say.
        """
        return {
            "intent":      "unknown",
            "config_code": None,
            "quantity":    None,
            "destination": None,
            "confidence":  0.0,
            "response":    f"Error del sistema: {reason}",
            "language":    "es",
            "error":       reason,
        }


def main(args=None):
    rclpy.init(args=args)
    node = CommandParserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._db_conn:
            node._db_conn.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
