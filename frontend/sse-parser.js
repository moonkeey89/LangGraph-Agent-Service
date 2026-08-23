/**
 * Incremental SSE parser with no DOM dependency.
 * Uint8Array chunks are decoded with one streaming TextDecoder, so a Chinese
 * UTF-8 code point may safely span multiple response.body.read() calls.
 */
export function createSseParser({ onEvent, onInvalidJson }) {
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finished = false;

  function emitBlock(block) {
    if (!block.trim()) {
      return;
    }

    let eventName = "message";
    const dataLines = [];

    for (const line of block.split(/\r\n|\n|\r/)) {
      if (!line || line.startsWith(":")) {
        continue;
      }

      const colonIndex = line.indexOf(":");
      const field = colonIndex === -1 ? line : line.slice(0, colonIndex);
      let value = colonIndex === -1 ? "" : line.slice(colonIndex + 1);
      if (value.startsWith(" ")) {
        value = value.slice(1);
      }

      if (field === "event") {
        eventName = value || "message";
      } else if (field === "data") {
        dataLines.push(value);
      }
    }

    if (dataLines.length === 0) {
      return;
    }

    const rawData = dataLines.join("\n");
    try {
      onEvent({ event: eventName, data: JSON.parse(rawData) });
    } catch (error) {
      onInvalidJson({
        event: eventName,
        rawData,
        message: error instanceof Error ? error.message : "Invalid JSON"
      });
    }
  }

  function drain(flushRemainder = false) {
    while (true) {
      const boundary = /\r\n\r\n|\n\n|\r\r/.exec(buffer);
      if (!boundary) {
        break;
      }
      emitBlock(buffer.slice(0, boundary.index));
      buffer = buffer.slice(boundary.index + boundary[0].length);
    }

    if (flushRemainder && buffer.trim()) {
      emitBlock(buffer);
      buffer = "";
    }
  }

  return {
    push(chunk) {
      if (finished) {
        throw new Error("SSE parser is already finished");
      }
      if (typeof chunk === "string") {
        buffer += chunk;
      } else {
        buffer += decoder.decode(chunk, { stream: true });
      }
      drain();
    },

    finish() {
      if (finished) {
        return;
      }
      finished = true;
      buffer += decoder.decode();
      drain(true);
    }
  };
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(`SSE parser self-test failed: ${message}`);
  }
}

function parserHarness() {
  const events = [];
  const errors = [];
  const parser = createSseParser({
    onEvent: (event) => events.push(event),
    onInvalidJson: (error) => errors.push(error)
  });
  return { parser, events, errors };
}

function findByteSequence(source, target) {
  outer: for (let index = 0; index <= source.length - target.length; index += 1) {
    for (let offset = 0; offset < target.length; offset += 1) {
      if (source[index + offset] !== target[offset]) {
        continue outer;
      }
    }
    return index;
  }
  return -1;
}

/** Run in the real browser on page load; no Node or DOM framework required. */
export function runSseParserSelfTests() {
  const passed = [];

  {
    const { parser, events } = parserHarness();
    parser.push('event: started\ndata: {"thread');
    assert(events.length === 0, "half event must stay buffered");
    parser.push('_id":"one"}\n\n');
    assert(events[0].data.thread_id === "one", "half event completion");
    passed.push("half-event-buffering");
  }

  {
    const { parser, events } = parserHarness();
    parser.push("event: progress\n");
    parser.push('data: {"node":"agent"}');
    parser.push("\n\n");
    assert(events.length === 1, "multiple chunks form one event");
    passed.push("multiple-chunks-one-event");
  }

  {
    const { parser, events } = parserHarness();
    parser.push(
      'event: token\ndata: {"content":"A"}\n\n' +
        'event: completed\ndata: {"answer":"A"}\n\n'
    );
    assert(events.length === 2, "one chunk contains multiple events");
    passed.push("one-chunk-multiple-events");
  }

  {
    const encoder = new TextEncoder();
    const wire = encoder.encode('event: token\ndata: {"content":"中文"}\n\n');
    const chinese = encoder.encode("中");
    const position = findByteSequence(wire, chinese);
    assert(position >= 0, "Chinese bytes found");
    const { parser, events } = parserHarness();
    parser.push(wire.slice(0, position + 1));
    parser.push(wire.slice(position + 1));
    parser.finish();
    assert(events[0].data.content === "中文", "UTF-8 split decoding");
    passed.push("utf8-cross-byte-boundary");
  }

  {
    const { parser, events, errors } = parserHarness();
    parser.push("event: token\ndata: {not-json}\n\n");
    parser.push('event: token\ndata: {"content":"still-alive"}\n\n');
    assert(errors.length === 1, "invalid JSON is reported");
    assert(events[0].data.content === "still-alive", "parser continues safely");
    passed.push("invalid-json-recovery");
  }

  return passed;
}
