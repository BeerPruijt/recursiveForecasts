from dataclasses import dataclass
from typing import List

@dataclass(repr=True)
class VariableSpec:
    """Specification for a variable transformation in time series analysis."""
    name: str
    diff_order: int
    log_transform: bool
    lag_order: int
    
    def to_list(self) -> List:
        """Convert back to original list format for backwards compatibility."""
        return [self.name, self.diff_order, self.log_transform, self.lag_order]
    
    def get_transformed_column_name(self) -> str:
        """Get the name of the transformed variable."""
        if self.log_transform:
            name = f"log({self.name})"
        else:
            name = self.name
        if self.diff_order > 0:
            name = f"{name}_d{self.diff_order}"
        if self.lag_order > 0:
            name = f"{name}_l{self.lag_order}"
        return name