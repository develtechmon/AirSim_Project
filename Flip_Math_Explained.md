# Drone Flips Explained - Like You're 10 Years Old! 🚁

## Part 1: Understanding Rotation Speed

### What is "rad/s"?

**Radians per second** = How fast something spins

Think of a pizza:
- A full pizza = 2π radians = 360 degrees (one complete circle)
- Half pizza = π radians = 180 degrees
- Quarter pizza = π/2 radians = 90 degrees

```
    🍕 Full Pizza = 2π radians = 360°
        ___
       /   \
      |  •  |    One complete spin!
       \___/
```

---

## Part 2: How Fast to Flip?

### The Basic Question:
**"How fast should the drone spin to do ONE complete flip?"**

Let's use simple numbers:

| What We Want | Math | Answer |
|--------------|------|--------|
| One full flip | 360 degrees | Need to spin 360° |
| How long? | Let's say 0.4 seconds | Nice and quick! |
| How fast to spin? | 360° ÷ 0.4 seconds | 900 degrees per second |

But we use **radians** instead of degrees, so:

| Step | Calculation | Result |
|------|-------------|--------|
| One full circle | 2π radians | ≈ 6.28 radians |
| Divided by time | 6.28 ÷ 0.4 seconds | **16 rad/s** |

**So 16 rad/s means: "Spin one full circle in 0.4 seconds"**

---

## Part 3: Simple Flips (One Direction)

### Roll = Barrel Roll (Like an Airplane)

Imagine the drone is a pencil rolling on a table:

```
START:    After 90°:   After 180°:  After 270°:  After 360°:
  🚁         🚁           🚁            🚁           🚁
  ||         ==           ||            ==           ||
Normal    On its side   Upside down  On other side  Normal!
```

### Pitch = Forward/Backward Flip

Like a front flip or backflip:

```
START:    After 90°:   After 180°:  After 270°:  After 360°:
  🚁         🚁           🚁            🚁           🚁
  ↑          →            ↓             ←            ↑
Facing up  Facing right Upside down  Facing left  Back up!
```

---

## Part 4: The Code - Simple Flips

### Right Flip (Roll Right)

```python
roll_rate = 16      # Spin at 16 rad/s
pitch_rate = 0      # Don't pitch
yaw_rate = 0        # Don't turn
duration = 0.39     # For about 0.4 seconds
```

**Table:**

| Axis | Value | What it does |
|------|-------|--------------|
| Roll | +16 | Spin RIGHT like a barrel roll |
| Pitch | 0 | No forward/backward |
| Yaw | 0 | No turning left/right |
| Time | 0.39s | Just long enough for one flip |

**Result:** Drone barrel rolls to the RIGHT ✅

---

### Left Flip (Roll Left)

```python
roll_rate = -16     # Spin OPPOSITE direction
pitch_rate = 0
yaw_rate = 0
duration = 0.39
```

**Table:**

| Axis | Value | What it does |
|------|-------|--------------|
| Roll | **-16** | Spin LEFT (negative = opposite) |
| Pitch | 0 | No forward/backward |
| Yaw | 0 | No turning |
| Time | 0.39s | One flip |

**Result:** Drone barrel rolls to the LEFT ✅

---

### Front Flip (Pitch Forward)

```python
roll_rate = 0
pitch_rate = 16     # Now we pitch!
yaw_rate = 0
duration = 0.39
```

**Table:**

| Axis | Value | What it does |
|------|-------|--------------|
| Roll | 0 | No barrel roll |
| Pitch | **+16** | Flip FORWARD like a gymnast |
| Yaw | 0 | No turning |
| Time | 0.39s | One flip |

**Result:** Drone does a front flip ✅

---

### Back Flip (Pitch Backward)

```python
roll_rate = 0
pitch_rate = -16    # Negative = backward
yaw_rate = 0
duration = 0.39
```

**Table:**

| Axis | Value | What it does |
|------|-------|--------------|
| Roll | 0 | No barrel roll |
| Pitch | **-16** | Flip BACKWARD |
| Yaw | 0 | No turning |
| Time | 0.39s | One flip |

**Result:** Drone does a back flip ✅

---

## Part 5: Diagonal Flips (The Tricky Part!)

### The Problem:
**What if we want to flip at an ANGLE? Like front-right?**

Let's think about walking:

```
If you walk 3 steps FORWARD and 0 steps RIGHT:
    Total distance = 3 steps

If you walk 3 steps FORWARD and 3 steps RIGHT:
    You go diagonal, but you travel MORE distance!
    
    |     /
    |    /
    |   /  ← You walked LONGER (about 4.2 steps)
    |  /
    | /
    |/
```

### Using Pythagoras (The Triangle Rule)

When you combine two directions, the total is:

**Total = √(Forward² + Right²)**

Example:
- Walk 3 forward + 3 right
- Total = √(3² + 3²) = √(9 + 9) = √18 = 4.24 steps

**Uh oh!** We wanted to travel only 3 steps total, but we traveled 4.24!

---

## Part 6: Fixing Diagonal Flips

### The Solution: Go Slower on Each Axis

If we want the TOTAL spin speed to be 16 rad/s when combining roll and pitch:

**Math:**
```
Total speed² = Roll speed² + Pitch speed²
16² = Roll² + Pitch²
256 = Roll² + Pitch²
```

If Roll and Pitch are EQUAL (for 45° diagonal):
```
256 = Roll² + Roll²
256 = 2 × Roll²
Roll² = 128
Roll = √128 = 11.3 rad/s
```

**So: `diagonal_rate = 16 ÷ √2 = 11.3 rad/s`**

---

## Part 7: Diagonal Flip Tables

### Front-Right Flip ↗

```python
roll_rate = 11.3    # Spin right (slower than 16)
pitch_rate = 11.3   # Flip forward (slower than 16)
duration = 0.39     # Same time
```

**The Magic:**

| Step | Calculation | Result |
|------|-------------|--------|
| Roll speed² | 11.3² = 128 | |
| Pitch speed² | 11.3² = 128 | |
| Total² | 128 + 128 = 256 | |
| Total speed | √256 = **16** | Perfect! |

**Table:**

| Axis | Value | What it does |
|------|-------|--------------|
| Roll | +11.3 | Spin right (medium speed) |
| Pitch | +11.3 | Flip forward (medium speed) |
| Yaw | 0 | No turning |
| Time | 0.39s | One flip |

**Result:** Drone flips at 45° angle (front-right) ✅

---

### Front-Left Flip ↖

**Table:**

| Axis | Value | What it does |
|------|-------|--------------|
| Roll | **-11.3** | Spin LEFT |
| Pitch | +11.3 | Flip FORWARD |
| Yaw | 0 | No turning |
| Time | 0.39s | One flip |

**Check the math:**
- Total = √((-11.3)² + (11.3)²) = √(128 + 128) = √256 = **16 rad/s** ✅

---

### Back-Right Flip ↘

**Table:**

| Axis | Value | What it does |
|------|-------|--------------|
| Roll | +11.3 | Spin RIGHT |
| Pitch | **-11.3** | Flip BACKWARD |
| Yaw | 0 | No turning |
| Time | 0.39s | One flip |

---

### Back-Left Flip ↙

**Table:**

| Axis | Value | What it does |
|------|-------|--------------|
| Roll | **-11.3** | Spin LEFT |
| Pitch | **-11.3** | Flip BACKWARD |
| Yaw | 0 | No turning |
| Time | 0.39s | One flip |

---

## Part 8: Complete Summary Table

### All 8 Directions!

| Flip Direction | Roll Rate | Pitch Rate | Total Speed | Why? |
|----------------|-----------|------------|-------------|------|
| **RIGHT** → | +16 | 0 | 16 rad/s | Only rolling, full speed |
| **FRONT-RIGHT** ↗ | +11.3 | +11.3 | 16 rad/s | √(11.3² + 11.3²) = 16 |
| **FRONT** ↑ | 0 | +16 | 16 rad/s | Only pitching, full speed |
| **FRONT-LEFT** ↖ | -11.3 | +11.3 | 16 rad/s | √(11.3² + 11.3²) = 16 |
| **LEFT** ← | -16 | 0 | 16 rad/s | Only rolling, full speed |
| **BACK-LEFT** ↙ | -11.3 | -11.3 | 16 rad/s | √(11.3² + 11.3²) = 16 |
| **BACK** ↓ | 0 | -16 | 16 rad/s | Only pitching, full speed |
| **BACK-RIGHT** ↘ | +11.3 | -11.3 | 16 rad/s | √(11.3² + 11.3²) = 16 |

**Notice:** They ALL end up at 16 rad/s total! That's the secret! 🎯

---

## Part 9: Why Does This Matter?

### Wrong Way (Don't Do This!)

```python
# Wrong: Use full speed for both
roll_rate = 16
pitch_rate = 16
```

**What happens:**
- Total speed = √(16² + 16²) = √512 = **22.6 rad/s**
- WAY TOO FAST!
- Drone spins 1.4 times instead of 1 time
- Ends up upside down 😱

### Right Way (Do This!)

```python
# Correct: Reduce each axis
diagonal_rate = 16 / √2 = 11.3
roll_rate = 11.3
pitch_rate = 11.3
```

**What happens:**
- Total speed = √(11.3² + 11.3²) = **16 rad/s**
- Perfect speed!
- Drone spins exactly 1 time
- Lands right-side up 😊

---

## Part 10: The Formula

### Remember This Forever:

```
For diagonal flips at 45°:

diagonal_rate = full_speed ÷ √2

Where √2 ≈ 1.414
```

**Why √2?**

Because when you walk equal distances in two directions:
- Total distance = √(distance² + distance²)
- Total distance = √(2 × distance²)
- Total distance = distance × √2

So to keep total the same, divide by √2!

---

## Part 11: Practice Problems!

### Problem 1:
If you want a flip speed of 20 rad/s, what should the diagonal rate be?

**Answer:**
```
diagonal_rate = 20 ÷ √2 = 20 ÷ 1.414 = 14.14 rad/s
```

**Check:** √(14.14² + 14.14²) = √400 = 20 rad/s ✅

---

### Problem 2:
You set roll = 12 and pitch = 12. What's the total spin speed?

**Answer:**
```
Total = √(12² + 12²) = √(144 + 144) = √288 = 16.97 rad/s
```

---

### Problem 3:
For a 30° angle flip (not 45°), if pitch = 16, what should roll be?

**This is HARD! Here's how:**

At 30° angle:
- sin(30°) = 0.5
- cos(30°) = 0.866

```
roll = 16 × sin(30°) = 16 × 0.5 = 8 rad/s
pitch = 16 × cos(30°) = 16 × 0.866 = 13.86 rad/s
```

**Check:** √(8² + 13.86²) = √(64 + 192) = √256 = 16 rad/s ✅

---

## Part 12: Real Numbers Used in Code

### Our Settings:

```python
base_rate = 16.0                    # Full speed
diagonal_rate = 16.0 / 1.414       # = 11.31 rad/s
flip_duration = (2 × π) / 16       # = 6.28 / 16 = 0.39 seconds
```

### What Actually Happens:

| Time | Angle Rotated | Status |
|------|---------------|--------|
| 0.00s | 0° | Starting |
| 0.10s | 90° | Quarter way |
| 0.20s | 180° | Upside down! |
| 0.29s | 270° | Three quarters |
| 0.39s | 360° | Complete! ✅ |

---

## Summary - The Big Picture

### Simple Flips (One Direction):
- Use **full speed** (16 rad/s) on ONE axis
- Other axis = 0
- Duration = 2π / 16 = 0.39 seconds

### Diagonal Flips (Two Directions):
- Use **reduced speed** (11.3 rad/s) on BOTH axes
- This makes total speed still = 16 rad/s
- Same duration = 0.39 seconds

### The Magic Formula:
```
diagonal_rate = full_speed ÷ √2
```

**That's it!** Now you understand drone flip math! 🎓✨

---

## Visual Summary

```
        FRONT (0, 16)
             ↑
    FrontLeft ↖  ↗ FrontRight
   (-11.3,11.3)   (11.3,11.3)
             |
LEFT ←-------•-------→ RIGHT
(-16,0)      |      (16,0)
             |
    BackLeft ↙  ↘ BackRight
   (-11.3,-11.3)  (11.3,-11.3)
             ↓
        BACK (0, -16)

All arrows have the same length = 16 rad/s!
```

**Now you're a flip expert!** 🚁🎯