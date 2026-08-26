"use strict";

let model = null;
let working = false;

const $ = (selector) => document.querySelector(selector);
const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");
const title = (value) => String(value).replaceAll("-", " ").replace(/\b\w/g, c => c.toUpperCase());

async function request(path, options = {}) {
  setWorking(true);
  try {
    const response = await fetch(path, options);
    const data = await response.json();
    if (!response.ok) {
      if (response.status === 409 && data.current) render(data.current);
      throw new Error(data.error || `Request failed (${response.status})`);
    }
    hideError();
    return data;
  } catch (error) {
    showError(error instanceof Error ? error.message : String(error));
    throw error;
  } finally {
    setWorking(false);
  }
}

function setWorking(value) {
  working = value;
  $("#busy").classList.toggle("hidden", !value);
  document.querySelectorAll("button").forEach(button => button.disabled = value || (button.id === "undo" && !model?.can_undo));
}
function showError(message) { $("#error").textContent = message; $("#error").classList.remove("hidden"); }
function hideError() { $("#error").classList.add("hidden"); }

function cardHtml(cardId, {compact = false, effectOrdinal = null} = {}) {
  const card = model.cards[cardId];
  if (!card) return `<div class="unknown-card">Unknown card</div>`;
  const dogma = compact ? "" : card.dogma.map((text, index) =>
    `<div class="dogma"${effectOrdinal === index + 1 ? ' style="color:var(--accent);font-weight:700"' : ""}>${escapeHtml(text)}</div>`
  ).join("");
  return `<article class="game-card ${card.color}">
    <strong>${escapeHtml(card.name)}</strong>
    <div class="meta">Age ${card.age} · ${escapeHtml(card.color)} · ${escapeHtml(card.featured_icon)}</div>
    <div class="icon-row">${card.icons.map(icon => `<span class="icon">${escapeHtml(icon)}</span>`).join("")}</div>
    ${dogma}
  </article>`;
}

function zoneHtml(name, zone) {
  const remainingAges = [...zone.values];
  const known = zone.known_cards.map(cardId => {
    const age = model.cards[cardId]?.age;
    const index = remainingAges.indexOf(age);
    if (index >= 0) remainingAges.splice(index, 1);
    return cardHtml(cardId, {compact: true});
  }).join("");
  const unknown = remainingAges.map(age => `<div class="unknown-card">Age ${age} card</div>`).join("");
  return `<section class="zone">
    <div class="zone-heading"><h3>${escapeHtml(name)}</h3><small>${zone.count ?? zone.values.length} card${zone.values.length === 1 ? "" : "s"}</small></div>
    <div class="card-list">${known}${unknown || (!known ? '<span class="muted">Empty</span>' : "")}</div>
  </section>`;
}

function stackHtml(stack) {
  const top = stack.top_card_id ? cardHtml(stack.top_card_id, {compact: true}) : '<div class="stack-empty">Empty</div>';
  const covered = stack.covered_cards || [];
  const identities = covered.filter(card => card.card_id).map(card => model.cards[card.card_id]?.name || card.card_id);
  const coveredCount = stack.covered_count === null ? "hidden" : stack.covered_count;
  const visibleIcons = covered.flatMap(card => card.visible_icons || []);
  return `<section class="stack ${stack.color}">
    <div class="stack-head"><h4>${escapeHtml(stack.color)}</h4><span class="badge">${escapeHtml(stack.splay)} splay</span></div>
    ${top}
    ${covered.length || stack.covered_count ? `<div class="covered">Covered: ${escapeHtml(coveredCount)}${identities.length ? ` · ${escapeHtml(identities.join(", "))}` : ""}</div>` : ""}
    ${visibleIcons.length ? `<div class="icon-row">Exposed: ${visibleIcons.map(icon => `<span class="icon">${escapeHtml(icon)}</span>`).join("")}</div>` : ""}
  </section>`;
}

function iconCounts(player) {
  const counts = {};
  for (const stack of player.board) {
    if (stack.top_card_id) {
      for (const icon of model.cards[stack.top_card_id].icons) counts[icon] = (counts[icon] || 0) + 1;
    }
    for (const covered of stack.covered_cards || []) {
      for (const icon of covered.visible_icons || []) counts[icon] = (counts[icon] || 0) + 1;
    }
  }
  return Object.entries(counts).sort().map(([icon, count]) => `<span class="icon">${escapeHtml(icon)} ${count}</span>`).join("");
}

function achievementsHtml(player) {
  const normal = player.normal_achievements.map(item => title(item));
  const special = player.special_achievements.map(item => title(item));
  const all = [...normal, ...special];
  return all.length ? all.map(item => `<span class="badge accent">${escapeHtml(item)}</span>`).join("") : '<span class="muted">None</span>';
}

function playerHtml(player, observation) {
  const active = observation.active_player === player.player_id;
  const achievementCount = player.normal_achievements.length + player.special_achievements.length;
  return `<section class="player-summary ${active ? "active" : ""}">
      <div><p class="eyebrow">${player.player_id === observation.viewer ? "Current view" : "Opponent"}</p><h2>${title(player.player_id)}</h2></div>
      <div class="badges">${active ? '<span class="badge accent">Active</span>' : ""}<span class="badge">${achievementCount} achievements</span></div>
    </section>
    ${zoneHtml("Hand", player.hand)}
    <section class="zone"><div class="zone-heading"><h3>Board</h3><small>visible icon totals</small></div><div class="icon-counts">${iconCounts(player) || '<span class="muted">No icons</span>'}</div><div class="board">${player.board.map(stackHtml).join("")}</div></section>
    ${zoneHtml("Score pile", player.score_pile)}
    <section class="zone"><div class="zone-heading"><h3>Achievements</h3><small>${achievementCount} claimed</small></div><div class="badges">${achievementsHtml(player)}</div></section>`;
}

function decisionHtml(decision) {
  if (!decision) return '<p class="muted">The game is complete. Download the log or start a new game.</p>';
  let instruction = decision.kind === "starting-meld" ? "Choose a starting meld in secret" :
    decision.kind === "turn-action" ? "Choose a paid action" : "Resolve the effect choice";
  const source = decision.source;
  let sourceHtml = "";
  if (source) {
    const ordinal = source.effect_id?.ordinal || null;
    sourceHtml = `<div class="source-card">${cardHtml(source.card_id, {effectOrdinal: ordinal})}</div>`;
    if (ordinal) instruction = `Resolve effect ${ordinal}`;
  }
  const context = decision.context;
  const chooserDetail = decision.chooser !== decision.executor ? ` · chooses for ${title(decision.executor)}` : "";
  const activatorDetail = decision.dogma_activator ? ` · activated by ${title(decision.dogma_activator)}` : "";
  const selectedCards = context?.selected_so_far?.length ? context.selected_so_far.map(cardId => model.cards[cardId]?.name || cardId).join(" → ") : null;
  const contextBadges = context ? [
    context.demand ? "Demand" : null,
    context.shared ? "Shared" : null,
    context.nested ? "Nested" : null,
    context.featured_icon ? `Featured: ${context.featured_icon}` : null,
    context.activator_icons !== null && context.activator_icons !== undefined ? `Icons ${context.activator_icons} vs ${context.opponent_icons}` : null,
    context.maximum_count > 1 ? `Choose ${context.minimum_count}–${context.maximum_count}` : null,
    context.incremental_selection && context.incremental_selection !== "none" ? title(context.incremental_selection) : null,
    selectedCards ? `Selected: ${selectedCards}` : null,
  ].filter(Boolean).map(text => `<span class="badge">${escapeHtml(text)}</span>`).join("") : "";
  const setupNote = model.pending_decision_count > 1 ? '<p class="muted">Both players have a pending secret setup choice. Player 1 goes first on this screen.</p>' : "";
  const actions = decision.legal_actions.map((item, index) => `<button class="action-button ${index === 0 ? "primary" : ""}" data-action="${index}"><span>${escapeHtml(item.label)}</span><small>${escapeHtml(item.payload.kind)}</small></button>`).join("");
  return `<p class="decision-kicker">${title(decision.chooser)} · ${escapeHtml(decision.kind)}${escapeHtml(chooserDetail)}${escapeHtml(activatorDetail)}</p>
    <h2>${escapeHtml(instruction)}</h2>
    ${setupNote}${sourceHtml}<div class="context">${contextBadges}</div>
    <div class="actions">${actions}</div>`;
}

function suppliesHtml(observation) {
  const supplies = observation.supplies.map(item => `<div class="supply"><span>Age ${item.age}</span><strong>${item.count}</strong></div>`).join("");
  const normals = observation.available_normal_achievements.map(item => `<li>${escapeHtml(title(item))}</li>`).join("");
  const specials = observation.available_special_achievements.map(item => {
    const info = model.special_achievements[item];
    return `<li title="${escapeHtml(info?.condition || "")}">${escapeHtml(info?.name || title(item))}</li>`;
  }).join("");
  const reveals = [
    ...observation.revealed_colors.map(color => `Revealed color: ${title(color)}`),
    ...observation.revealed_cards.map(cardId => `Revealed card: ${model.cards[cardId]?.name || cardId}`),
  ];
  return `<h3>Supply</h3>${reveals.length ? `<div class="context">${reveals.map(item => `<span class="badge accent">${escapeHtml(item)}</span>`).join("")}</div>` : ""}<div class="supplies-grid">${supplies}</div>
    <div class="achievements"><strong>Available achievements</strong><div class="badges"><span class="badge">${observation.available_normal_achievements.length} normal</span><span class="badge">${observation.available_special_achievements.length} special</span></div>
    <details><summary>Show achievement list</summary><ul>${normals}${specials}</ul></details></div>`;
}

function render(nextModel) {
  model = nextModel;
  const obs = model.observation;
  $("#seed").value = model.seed;
  $("#undo").disabled = working || !model.can_undo;
  $("#status").innerHTML = `<span>Phase <strong>${escapeHtml(title(model.phase))}</strong></span><span>Turn <strong>${obs.turn_number}</strong></span><span>Active <strong>${obs.active_player ? escapeHtml(title(obs.active_player)) : "—"}</strong></span><span>Paid actions <strong>${obs.paid_actions_remaining}</strong></span><span>Moves <strong>${model.transition_count}</strong></span><span title="${escapeHtml(model.state_hash)}">State <strong>${escapeHtml(model.state_hash.slice(7, 15))}</strong></span>`;
  $("#player-1").innerHTML = playerHtml(obs.players.find(player => player.player_id === "player-1"), obs);
  $("#player-2").innerHTML = playerHtml(obs.players.find(player => player.player_id === "player-2"), obs);
  $("#decision").innerHTML = decisionHtml(model.decision);
  $("#supplies").innerHTML = suppliesHtml(obs);
  $("#history").innerHTML = model.history.map(item => `<li><strong>${escapeHtml(item.label)}</strong> <span class="muted">${escapeHtml(item.kind)}</span></li>`).join("") || '<li class="muted">No actions yet</li>';

  const terminal = $("#terminal");
  if (model.terminal_result) {
    const winners = model.terminal_result.winners.length ? model.terminal_result.winners.map(title).join(" & ") : "Draw";
    terminal.innerHTML = `<h2>${escapeHtml(winners)}</h2><p>Game ended: ${escapeHtml(title(model.terminal_result.reason))}</p>`;
    terminal.classList.remove("hidden");
  } else {
    terminal.classList.add("hidden");
  }

  document.querySelectorAll("[data-action]").forEach(button => button.addEventListener("click", async () => {
    if (working) return;
    const item = model.decision.legal_actions[Number(button.dataset.action)];
    const next = await request("/api/action", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action: item.payload, game_id: model.game_id, revision: model.revision}),
    });
    render(next);
  }));
}

$("#new-game").addEventListener("click", async () => {
  const seed = Number($("#seed").value);
  if (!Number.isSafeInteger(seed)) return showError("Seed must be an integer.");
  if (model?.transition_count && !confirm("Start a new game and discard the current in-memory game?")) return;
  render(await request("/api/new", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({seed})}));
});
$("#undo").addEventListener("click", async () => {
  if (!model?.can_undo || working) return;
  render(await request("/api/undo", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({game_id: model.game_id, revision: model.revision})}));
});

request("/api/state").then(render).catch(() => {});
