import parsem5

def get_commits(stats_path):
    """Get the commits section of a stats file"""
    stats = parsem5.parse(stats_path)

    # Get "processor.core" section of all periods simulated
    processors = [s["board"]["processor"]["start"]["core"] for s in stats]

    # Get committed instructions stats
    instructions = [p["commitStats0"] for p in processors]
    return instructions
    #instructions_fields = instructions[0].keys()
    #alu_fields = [key for key in instructions_fields if "committedInstType" in key]
    #control_fields = [key for key in instructions_fields if "committedControl" in key]
    #fields = alu_fields+["committedControl::IsControl"]
