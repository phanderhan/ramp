// Delegate to the document so mode changes and reruns need no rebinding.
// Streamlit exposes each keyed widget with a stable st-key-<key> class.
export default function ({ data: sliderKeys, setTriggerValue }) {
  function resetOnDoubleClick(event) {
    if (!(event.target instanceof Element)) return;
    for (const key of sliderKeys) {
      const widget = event.target.closest(`.st-key-${key}`);
      if (!widget || !widget.querySelector('[role="slider"]')) continue;
      event.preventDefault();
      event.stopPropagation();
      setTriggerValue("reset", key);
      return;
    }
  }

  const options = { capture: true };
  document.addEventListener("dblclick", resetOnDoubleClick, options);
  return () => document.removeEventListener("dblclick", resetOnDoubleClick, options);
}
