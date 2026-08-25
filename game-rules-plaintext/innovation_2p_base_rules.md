# Innovation — Two-Player Base-Game Rules

**Edition:** Innovation Base, Third Edition  
**Scope:** Two players, base game only. Individual card text is not reproduced here and remains part of the rules.

## 1. Objective

A player wins immediately upon claiming **six achievements**. Normal and special achievements both count.

If the game instead ends because a draw would require a card higher than age 10, the player with the higher score wins. Score ties are broken by number of achievements; if that is also tied, the game is a draw.

Some card effects can also end the game or cause a player to win immediately.

## 2. Cards, Board, and Score

The game contains 105 unique innovation cards divided among ages 1 through 10.

Each innovation card has:

- A unique title.
- A value equal to its age.
- One of five colors.
- Four potential icon positions. Three contain functional icons and one contains a card image; the card image has no game effect.
- One or more dogma effects.
- A featured icon used to determine sharing and immunity to demands.

There are six icon types: **castle, crown, leaf, lightbulb, factory, and clock**.

Each player has:

- A **hand**.
- A **board**, containing up to five color stacks.
- A face-down **score pile**.
- Any achievements they have claimed.

The completely visible card at the top of each color stack is its **top card**. Only top cards can be chosen for a Dogma action. Covered cards can still contribute icons when their color is splayed.

A player's score is the sum of the values of all cards in their score pile.

## 3. Setup

1. Separate the innovation cards into ten face-down supply piles by age, 1 through 10.
2. Randomly and without looking, set aside one card from each pile of ages 1 through 9. These nine cards are the normal achievements.
3. Place the five special achievements nearby.
4. Each player draws two age-1 cards.
5. Both players simultaneously choose one of those cards to place face-up as the first card on their board. The unchosen card becomes that player's initial hand.
6. Compare the titles of the two starting board cards. The card whose title comes first alphabetically determines the first player.
7. The first player takes only one action on their first turn. The other player's first turn, and every later turn, consists of two actions.

## 4. Turn Structure

On a normal turn, the active player must take exactly two actions.

The four action types are:

1. **Draw**
2. **Meld**
3. **Dogma**
4. **Achieve**

The actions may be taken in any order, and the same action may be taken twice.

A free Draw gained from sharing is not one of the turn's two actions.

## 5. Draw Action

To take a Draw action:

1. Find the highest-valued top card on your board.
2. Draw the top card from the supply pile of that age.
3. Add it to your hand.

If the required supply pile is empty, draw from the next higher non-empty pile. Never fall back to a lower age. This rule applies both to Draw actions and to card effects that cause draws.

If satisfying a draw would require a card higher than age 10, the game ends immediately.

The value of something you do not have is treated as 0. This applies, for example, when no top card of a specified color exists.

## 6. Meld Action

Choose any card from your hand and place it on top of the stack of its matching color on your board.

- If no stack of that color exists, the card starts a new stack.
- The card can be melded regardless of how its value compares with the existing top card.
- If the color is already splayed, the newly melded card continues the same splay direction.

## 7. Achieve Action

There are nine normal achievements, one from each age 1 through 9.

To claim the normal achievement of age `N`, both conditions must be true:

- Your score is at least `5 × N`.
- You have at least one top card of value `N` or higher.

Claiming an achievement does not spend or remove points. Once claimed, an achievement cannot be taken away.

Achievements do not have to be claimed in age order. Once a normal achievement has been claimed, it is not replenished.

The identity of a normal achievement card is never revealed, even after it is claimed.

## 8. Dogma Action

To take a Dogma action, choose one of your top cards and execute all of its dogma effects in printed order.

### 8.1 Determine Sharing and Demand Immunity

At the start of the Dogma action:

1. Identify the activated card's featured icon.
2. Count how many visible copies of that icon each player has.
3. Keep those counts fixed for the entire Dogma action. Do not recount between effects.

For the opponent:

- If they have **at least as many** featured icons as the active player, they share non-demand effects and ignore demand effects.
- If they have **fewer** featured icons, they do not share non-demand effects and must execute demand effects.

Equality therefore allows sharing and grants immunity to demands.

### 8.2 Non-Demand Effects

A non-demand effect is any dogma effect that does not begin with “I Demand.”

For each non-demand effect:

1. If the opponent is eligible to share, the opponent executes the effect first.
2. The active player then executes the effect.
3. Complete that effect fully before moving to the next printed effect.

Each player makes any choices required while executing the effect for themselves.

An effect marked **“execute for yourself only”** is not shared, regardless of icon counts. It can still affect the opponent if its instructions say so.

### 8.3 Demand Effects

A demand effect begins with “I Demand.”

The opponent executes a demand only if they had fewer of the featured icon at the start of the Dogma action. An opponent with at least as many ignores it.

In demand text:

- “I” and “my” refer to the player who activated the card.
- “You” and “your” refer to the opponent executing the demand.

The activating player does not execute the demand against themselves.

### 8.4 Sharing Bonus

After the entire Dogma action is complete, the active player takes **one free Draw action** if the opponent's execution of at least one shared non-demand effect caused the game to change.

A game change includes one or more cards being:

- Drawn.
- Revealed.
- Melded.
- Tucked.
- Splayed.
- Scored.
- Exchanged.
- Otherwise moved.

Only one free Draw is gained, no matter how many effects were shared.

If the opponent follows a shared effect but nothing in the game changes, no free Draw is gained.

### 8.5 Partial Execution

If an effect cannot be completed in full, perform every part that can be performed and ignore only the impossible remainder.

Examples:

- A demand for three cards takes two if only two eligible cards exist.
- An exchange still occurs when one of the two locations is empty.

It is legal to take a Dogma action even when none of the activated card's effects will change the game.

## 9. Splaying and Visible Icons

A color stack is always in one of four states:

- Unsplayed.
- Splayed left.
- Splayed right.
- Splayed up.

The top card remains completely visible. Covered cards expose icons according to the splay direction:

| Stack state | Functional positions visible on each covered card |
|---|---|
| Unsplayed | None |
| Splayed left | Bottom-right |
| Splayed right | Top-left and bottom-left |
| Splayed up | Bottom-left, bottom-center, and bottom-right |

If an exposed position contains the card image instead of an icon, it contributes nothing.

To splay a color, spread all cards in the indicated direction. A left splay reveals one position on each covered card, a right splay reveals two, and an up splay reveals three.

If a stack is already splayed and is splayed in a different direction, replace the old splay with the new one.

A stack containing zero or one cards is always unsplayed. If a splayed stack is reduced to zero or one cards, it loses its splay direction and does not remember it if cards are later added.

Melding or tucking a card into an existing splayed stack preserves its current splay direction.

## 10. Special Achievements

Special achievements are claimed immediately when their conditions are met. They do not require an Achieve action.

Once a special achievement is claimed, it cannot be taken away.

If both players become eligible for the same special achievement at exactly the same time, the current player claims it.

### Monument

Claim Monument immediately if you:

- Tuck at least six cards during one turn, **or**
- Score at least six cards during one turn.

Tucks and scores are separate conditions and are not added together.

Cards transferred from another player into your score pile do not count. Cards entering your score pile through an exchange also do not count.

Monument may also be claimed through **Masonry**.

### Empire

Claim Empire immediately if you have at least three visible icons of every one of the six icon types.

Empire may also be claimed through **Construction**.

### World

Claim World immediately if you have at least twelve visible clock icons on your board.

World may also be claimed through **Translation**.

### Wonder

Claim Wonder immediately if:

- You have all five colors on your board, and
- Every color is splayed either right or up.

Wonder may also be claimed through **Invention**.

### Universe

Claim Universe immediately if:

- You have five top cards, one of each color, and
- Every top card has value 8 or higher.

Universe may also be claimed through **Astronomy**.

## 11. Rules Keywords

### Bottom

The bottom card is the card at the bottom of a color stack. In a one-card stack, that card is both top and bottom.

### Card Image

One of a card's four potential icon positions contains an image of the innovation. It is not an icon and provides no game benefit.

### Draw and X

For “draw and meld,” “draw and tuck,” or “draw and score”:

1. Draw the requested card, applying the empty-pile rule.
2. Immediately perform the second operation using that exact card.

### Exchange

Swap the cards in the two specified locations. The exchange still occurs if one location is empty.

Cards entering a score pile through an exchange do not count as scored for Monument.

### Execute for Yourself Only

Do not share the effect, regardless of icon counts. The effect can still affect the opponent according to its text.

### Highest and Lowest

“Highest” and “lowest” refer to card value, which is the card's age. If multiple eligible cards are tied, the player who owns or controls the choice selects which tied card is affected.

### Non-Demand Effect

Any dogma effect that does not begin with “I Demand.”

### Remove

A removed card is set aside outside the game. Fission can remove all cards in both players' hands, score piles, and boards. Achievements remain.

### Return

Place the returned card face-down at the bottom of the supply pile matching its age.

If multiple cards are returned at once, the player returning them chooses their order. Returning a card can recreate a supply pile that was empty.

### Score

Place the card face-down in your score pile. Its point value equals its age.

### Top

The top card of a color stack is its completely visible card.

### Tuck

Place the card at the bottom of its matching-color stack, preserving an existing splay when possible. If that color is absent, the card forms a new one-card stack.

### Value

A card's value is its age. The value of something absent is 0.

### Win and Game-Ending Effects

A card effect that says a player wins ends the game immediately if its condition is satisfied.

An effect saying that “the single player with the most X wins” ends the game only if exactly one player has the most X. If the players are tied, ignore the entire win effect and continue the game.

## 12. Winning and Game End

### Achievement Victory

A player wins immediately upon obtaining their sixth achievement. Normal and special achievements both count.

### Drawing Beyond Age 10

If a Draw action or card effect would require a card higher than age 10:

1. End the game immediately.
2. The player with the higher current score wins.
3. If scores are tied, the player with more achievements wins.
4. If achievements are also tied, the game is a draw.

### Card-Effect Victory

Card effects that directly award victory or end the game take effect immediately according to their text and the general win-effect rules above.

## 13. Information Rules

The following are hard rules:

- Neither player may ever look at the identity of a normal achievement card, even after it has been claimed.
- A player may always inspect every card in their own hand, score pile, and board, including their own covered cards.
- Both players may always count and see the values of cards in every supply pile, each hand, and each score pile.

The group must decide whether these are public:

- The identities of partially covered cards on the opponent's board.
- The number of cards in an opponent's unsplayed color stack.

The rulebook's stated convention is that neither is public, but either information policy is legal.

## 14. Official Card-Specific Clarifications

These clarifications apply when the corresponding cards are implemented:

- **Road Building:** You may choose to meld only one card even if two or more eligible cards are in your hand.
- **Machinery:** All applicable exchanges occur, even if a player would prefer not to make one of them.
- **Anatomy:** The victim does not have to choose a score-pile card that causes a top card to be returned unless every eligible choice would do so.
- **Measurement:** If every possible color is a one-card stack and therefore cannot actually become splayed, the player may still choose one and draw the specified 1.
- **Democracy:** Resolve players sequentially using the state produced by earlier executions of the effect.
- **Publication:** A rearranged stack retains its previous splay direction.
- **Fission:** All cards in hands, score piles, and boards can be removed. Achievements remain, and play can resume using the cards left in the supply.
- **General rule:** If an effect cannot be completed in full, do everything possible and ignore the rest.
