class AbstractTool:
    """
    Abstract class for defining a tool that can be used by an AI agent.
    """
    name: str
    description: str
    parameters: dict[str, dict[str, str]]

    def __call__(self, *args, **kwargs):
        """
            arguments have to be described in tool parameters
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def to_dict(self) -> dict:
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
            },
            "required": list(self.parameters.keys())
        }
