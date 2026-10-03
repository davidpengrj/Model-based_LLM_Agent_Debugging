'use strict';

(() => {
  let initialized = false;

  function initialize() {
    if (initialized || !window.portfolioControls) return;
    initialized = true;

    const controls = window.portfolioControls;
    const play = document.getElementById('trace-play');
    const label = document.getElementById('trace-play-label');
    const icon = play.querySelector('.play-icon');
    const previous = document.getElementById('trace-prev');
    const next = document.getElementById('trace-next');
    const scrubber = document.getElementById('trace-scrubber');
    const position = document.getElementById('playback-position');
    const status = document.getElementById('playback-status');
    const wrapper = play.closest('.trace-playback') || play.parentElement;
    const count = controls.getCount();
    let timer = null;
    let playing = false;
    let hasPlayed = false;
    let statusKey = 'ready';

    const messages = {
      play: ['Play the trace', '播放真实轨迹'],
      pause: ['Pause the trace', '暂停播放'],
      replay: ['Replay the trace', '重新播放轨迹'],
      previous: ['Previous event', '上一个事件'],
      next: ['Next event', '下一个事件'],
      ready: ['Choose an event or play the full trace.', '选择一个事件，或播放整条真实轨迹。'],
      playing: ['Playing the recorded trace. Pause to inspect any event.', '正在播放真实记录。随时暂停，查看当前事件。'],
      paused: ['Playback paused. Inspect the evidence or continue.', '已暂停。可以查看证据，也可以继续播放。'],
      complete: ['Trace complete. Replay or inspect an event.', '轨迹播放完毕。可以重播，或选择事件查看。'],
      selected: ['Event selected. Inspect its recorded evidence.', '已选择事件，可以查看对应的真实证据。'],
      hidden: ['Playback paused while this page was hidden.', '页面隐藏时已暂停播放。'],
    };
    const isChinese = () => document.documentElement.lang.startsWith('zh');
    const text = key => messages[key][isChinese() ? 1 : 0];
    const eventLabel = index => isChinese()
      ? `事件 ${String(index + 1).padStart(2, '0')} / ${String(count).padStart(2, '0')}`
      : `Event ${String(index + 1).padStart(2, '0')} / ${String(count).padStart(2, '0')}`;

    scrubber.min = '0';
    scrubber.max = String(count - 1);
    scrubber.step = '1';
    play.disabled = false;
    scrubber.disabled = false;

    function synchronize() {
      const index = controls.getEvent();
      const labelKey = playing ? 'pause' : hasPlayed && index === count - 1 ? 'replay' : 'play';
      label.textContent = text(labelKey);
      play.setAttribute('aria-label', text(labelKey));
      play.setAttribute('aria-pressed', String(playing));
      icon.textContent = playing ? 'Ⅱ' : '▶';
      wrapper.classList.toggle('is-playing', playing);
      previous.setAttribute('aria-label', text('previous'));
      next.setAttribute('aria-label', text('next'));
      previous.disabled = index === 0;
      next.disabled = index === count - 1;
      scrubber.value = String(index);
      scrubber.setAttribute('aria-valuetext', eventLabel(index));
      position.textContent = eventLabel(index);
      const announcement = text(statusKey);
      if (status.textContent !== announcement) status.textContent = announcement;
    }

    function cancelTimer() {
      if (timer !== null) window.clearTimeout(timer);
      timer = null;
    }

    function pause(reason = 'paused') {
      cancelTimer();
      playing = false;
      statusKey = reason;
      synchronize();
    }

    function scheduleNext() {
      cancelTimer();
      if (!playing) return;
      timer = window.setTimeout(() => {
        timer = null;
        if (!playing) return;
        const index = controls.getEvent();
        if (index >= count - 1) {
          pause('complete');
          return;
        }
        controls.selectEvent(index + 1, 'playback');
        if (index + 1 === count - 1) pause('complete');
        else scheduleNext();
      }, 2200);
    }

    play.addEventListener('click', () => {
      if (playing) {
        pause();
        return;
      }
      const restart = !hasPlayed || controls.getEvent() === count - 1;
      hasPlayed = true;
      playing = true;
      statusKey = 'playing';
      if (restart) controls.selectEvent(0, 'playback');
      synchronize();
      scheduleNext();
    });
    previous.addEventListener('click', () => controls.selectEvent(Math.max(0, controls.getEvent() - 1), 'previous'));
    next.addEventListener('click', () => controls.selectEvent(Math.min(count - 1, controls.getEvent() + 1), 'next'));
    scrubber.addEventListener('input', () => controls.selectEvent(Number(scrubber.value), 'scrub'));

    document.addEventListener('portfolio:event', event => {
      if (event.detail.source !== 'playback') pause('selected');
      else synchronize();
    });
    document.addEventListener('portfolio:language', synchronize);
    document.addEventListener('visibilitychange', () => {
      if (document.hidden && playing) pause('hidden');
    });
    window.addEventListener('pagehide', () => pause('hidden'));
    synchronize();
  }

  if (window.portfolioControls) initialize();
  else document.addEventListener('portfolio:ready', initialize, {once: true});
})();
