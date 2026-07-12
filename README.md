# Titanic Loss Playground

One tiny model — a single sigmoid neuron (logistic regression) — learns to predict who survived the Titanic, trained live in the browser on 160 real passengers. The same model is trained twice on the same data: once with the **quadratic loss (MSE)** and once with **cross-entropy**. The point of the demo is to see, not just be told, why cross-entropy is the standard loss for classification: its learning signal is biggest exactly when the model is confidently wrong, while the quadratic signal fades to almost nothing there.

## Try it

- Open `index.html` in a browser — one file, no installation, no build step, no API key, works offline.

## Using it in class

- Press **Play** with the default *Neutral* start: both models learn the "women and upper classes first" rule quickly; compare the loss-curve shapes and watch the six passenger cards update live (pause and **Step 1 epoch** to slow the story down).
- Switch the start to **Confidently wrong** and play again: the cross-entropy model recovers in ~20 epochs, the quadratic one needs ~290 — the "Why cross-entropy learns faster" chart explains the vanishing gradient in one picture.
- Point at Miss Helen Allison (a two-year-old in first class who did not survive): every reasonable model gets her wrong — real data has exceptions, and 100% accuracy is not the goal.

## How it works

The model combines four facts about each passenger — sex, age, ticket class, and (log-)fare, each standardised — into one number, then squashes it into a probability with the sigmoid function. Training is plain full-batch gradient descent (learning rate 0.5): after each epoch every weight moves a small step in the direction that shrinks the loss.

The only difference between the two models is which loss they shrink. The quadratic loss (p − y)² sends a learning push of 2(p − y)·p·(1 − p) back through the neuron; that extra p·(1 − p) factor is nearly zero whenever the prediction is close to 0 or 1 — including when it is confidently **wrong**, which is exactly when the model most needs correcting. Cross-entropy's push is simply p − y: biggest when the model is most wrong. On this data, from a deliberately backwards starting point, that makes cross-entropy about 15× faster to recover (see `verify.js`).

All numbers on screen are computed live — real forward passes, real losses, real gradient descent, nothing pre-recorded.

## Data

The embedded 160 passengers are a stratified sample of the public Kaggle/Vanderbilt Titanic training set (891 passengers), drawn so that survival rates by sex and class match the full data (subset vs full: women 73.7% vs 74.2%, men 18.4% vs 18.9%, overall 38.1% vs 38.4%). Only passengers with a recorded age were sampled. Fields kept: name, sex, age, class, fare, survived.

## Checking the maths

`node verify.js` extracts the core maths block straight out of `index.html` (so what is tested is exactly what runs in the browser) and checks that: analytic gradients match numerical finite differences for both losses; both models converge on the embedded data; cross-entropy reaches the target accuracy in fewer epochs than MSE (19 vs 291 from the confidently-wrong start); and the MSE gradient really does vanish on confident-wrong predictions (~32× smaller).

## Course context

Part of the *2 MA LMT :: Artificial Intelligence* course at the Faculty of English, AMU Poznań. Used in the loss-functions sequence (quadratic vs cross-entropy, Class 10). Rebuilt in 2026 from a vibe-coded demo used in the 2025/26 run.

## Licence

MIT — see [LICENSE](LICENSE).
