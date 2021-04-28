import munch

def LogOptions():
    lo = munch.Munch()
    # Save images each x iters
    lo.display_freq = 10000

    # Print info each x iters
    lo.print_freq = 10
    return lo
