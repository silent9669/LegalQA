class ProvisionMemory:
    """
    Provision-level memory lookup.
    Given document number and article, retrieves known training QA examples
    grounded in that specific legal provision to provide stylistic/substantive demonstrations.
    """
    def __init__(self, provision_dict: dict):
        self.store = provision_dict

    def lookup(self, doc_number: str, article: str) -> list[dict]:
        key = f"{doc_number}::{article}".strip()
        return self.store.get(key, [])
