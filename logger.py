def log(type: str, rank: int, value: any) -> None:
    """
    Custom logger function used for metrics purposes.
    """
    print(f"{type} {rank} {value}")
