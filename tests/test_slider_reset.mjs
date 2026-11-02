// Focused frontend handler checks without requiring a browser or dependencies.
// Run with: node --test tests/test_slider_reset.mjs
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../apps/slider_reset.js', import.meta.url), 'utf8');
const { default: mount } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('delegates double-clicks, ignores other controls, and cleans up on rerender', () => {
  globalThis.document = new EventTarget();
  globalThis.Element = class {};
  const keys = ['duration', 'shape_exponent', 'interval_shape_exponent', 'skew', 'tempo_bpm', 'cowbell_gain'];
  const calls = [];
  const data = { data: keys, setTriggerValue: (...args) => calls.push(args) };
  const cleanup = mount(data);

  function dispatch(type, key, isSlider = true) {
    const target = new Element();
    // Model a descendant of Streamlit's keyed widget wrapper.
    target.closest = (selector) => selector === `.st-key-${key}` ? {
      querySelector: (query) => query === '[role="slider"]' && isSlider ? {} : null,
    } : null;
    const event = new Event(type, { cancelable: true });
    Object.defineProperty(event, 'target', { value: target });
    document.dispatchEvent(event);
    return event;
  }

  for (const key of keys) {
    const before = calls.length;
    dispatch('click', key);
    assert.equal(calls.length, before, 'single clicks must not reset');
    assert.equal(dispatch('dblclick', key).defaultPrevented, true);
    assert.deepEqual(calls.at(-1), ['reset', key]);
    assert.equal(calls.length, before + 1);
  }
  const count = calls.length;
  dispatch('dblclick', 'start_rate_text');
  dispatch('dblclick', 'skew', false);
  assert.equal(calls.length, count, 'unrelated/non-slider controls must be ignored');

  cleanup();
  dispatch('dblclick', 'duration');
  assert.equal(calls.length, count, 'unmounted listener must be removed');
  const cleanupAgain = mount(data);
  dispatch('dblclick', 'interval_shape_exponent');
  dispatch('dblclick', 'interval_shape_exponent');
  assert.equal(calls.length, count + 2, 'remounts and repeated resets fire exactly once');
  cleanupAgain();
});
