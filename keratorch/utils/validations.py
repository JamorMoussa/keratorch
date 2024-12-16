from dataclasses import fields
from typing import Any 


__all__ = ["TypeChecker", ]


class TypeChecker:

    def __setattr__(self, name: str, value: Any, from_update: bool = False):

        # if not from_update:
        #     raise AttributeError(f"You must update the value of the attribute '{name}' by using 'update' method.")

        for field_info in fields(self):
            fname, ftype = field_info.name, field_info.type

            if field_info.name == name:

                if hasattr(ftype, '__origin__') and ftype.__origin__ is list:
                    if not isinstance(value, list):
                        raise TypeError(
                            f"'{self.__class__.__name__}.{fname}' must be a list, "
                            f"but got {type(value).__name__}."
                        )
                    
                    elif ftype.__args__ and not all(isinstance(v, ftype.__args__[0]) or v is None for v in value):
                        raise TypeError(
                            f"'{self.__class__.__name__}.{fname}' must be a list of {ftype.__args__[0].__name__}, "
                            f"but got a list with elements of type {[type(v).__name__ for v in value]}."
                        )
                    
                    elif len(value) != 2:
                        raise ValueError(
                            f"The 'value' must be a list of two elements (inputs, targets), but got {len(value)} instead."
                        )

                elif not isinstance(value, ftype) and value is not None:
                    raise TypeError(
                        f"'{self.__class__.__name__}.{fname}' must be of type {ftype.__name__}, "
                        f"but got {type(value).__name__}."
                    )
                
        super().__setattr__(name, value)

    def update(self, **kwargs):

        names = list(map(lambda field_info: field_info.name, fields(self)))

        for name, value in kwargs.items():
            if name in names:
                self.__setattr__(name, value, from_update=True)
            else: 
                raise KeyError(f"Attributes '{name}' is not supported in '{self.__class__.__name__}' dataclass.")
