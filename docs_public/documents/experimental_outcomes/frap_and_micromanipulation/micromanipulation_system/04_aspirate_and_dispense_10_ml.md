# MP285 Tool Interaction Record: Quickly Aspirate and Dispense 10 ml

## User Request

> Known conditions: The first position of the array is at A(0, 0); the array is 5x5, with 1 mm spacing between adjacent positions along X and 1 mm spacing between rows. The green ball reservoir is at B(500 mm, 500 mm) and the red ball reservoir is at C(-500 mm, -500 mm).
>
> Control the pump in the micromanipulation system to quickly aspirate/dispense 10 ml of liquid at a pump velocity of 100 uL/s.

## Planner Output

- **Module:** `mp285_controller`
- **Command:**

> Connect to both the MP-285 micromanipulator and the pump using default serial settings.; 
#Pump set velocity: Set the pump fluid handling speed to 100 uL/s.; 
#Pump in: Aspirate 10 ml of liquid at the configured pump velocity of 100 uL/s.; 
#Pump out: Dispense 10 ml of liquid at the configured pump velocity of 100 uL/s.

## User Confirmation

> confirm

## Executor-Generated Code

```python
volume_ml = 10.0
volume_ul = volume_ml * 1000.0
velocity_ul_per_s = 100.0

if velocity_ul_per_s <= 0:
    say("[ERROR] Pump velocity must be positive.")
    raise RuntimeError("Invalid pump velocity.")

if volume_ul <= 0:
    say("[ERROR] Pump volume must be positive.")
    raise RuntimeError("Invalid pump volume.")

say("[ACTION] Connecting to MP-285 micromanipulator and pump using default serial settings.")
connect()

say("[ACTION] Setting pump velocity to 100 uL/s.")
pump_set_velocity(velocity_ul_per_s)

say("[ACTION] Aspirating 10 ml (10000 uL) of liquid.")
pump_in(volume_ul)

say("[ACTION] Dispensing 10 ml (10000 uL) of liquid.")
pump_out(volume_ul)

say("[INFO] Pump aspiration and dispensing sequence completed.")
```
