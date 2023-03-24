"""

Metadata filtering

Support metadata filtering similar to mongodb

"""


def filter_item(filter_dict : dict, metadata_dict : dict) -> bool:
    def apply_operator(operator, key, value):
        if metadata_dict is None:
            return False        
        metadata_value = metadata_dict.get(key)
        if operator == "$eq":
            if isinstance(metadata_value, list):
                return value in metadata_value
            else:
                return metadata_value == value
        elif operator == "$ne":
            return metadata_value != value
        elif operator == "$gt":
            return metadata_value > value
        elif operator == "$gte":
            return metadata_value >= value
        elif operator == "$lt":
            return metadata_value < value
        elif operator == "$lte":
            return metadata_value <= value
        elif operator == "$in":
            if isinstance(metadata_value, list):
                return any(item in metadata_value for item in value)
            else:
                return metadata_value in value
        elif operator == "$nin":
            return all(item not in value for item in metadata_value) if isinstance(metadata_value, list) else metadata_value not in value
        elif operator == "$and":
            return all(apply_filter(sub_filter) for sub_filter in value)
        elif operator == "$or":
            return any(apply_filter(sub_filter) for sub_filter in value)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

    def apply_filter(filter_rule):
        if isinstance(filter_rule, dict):
            if all(isinstance(value, dict) for value in filter_rule.values()):
                return all(
                    apply_operator(op_key, filter_key, filter_rule[filter_key][op_key])
                    for filter_key in filter_rule
                    for op_key in filter_rule[filter_key]
                )
            elif "$and" in filter_rule:
                return all(apply_filter(sub_filter) for sub_filter in filter_rule["$and"])
            elif "$or" in filter_rule:
                return any(apply_filter(sub_filter) for sub_filter in filter_rule["$or"])
            else:
                return all(apply_operator("$eq", key, value) for key, value in filter_rule.items())
        else:
            raise ValueError(f"Invalid filter rule: {filter_rule}")

    return apply_filter(filter_dict)




"""
# alternative version but but this appears to be slower than the first one since it builds the dictionary of functions for each operator each time it is called
def filter_item(filter_dict, metadata_dict):
    def apply_operator(operator, key, value):
        metadata_value = metadata_dict.get(key)
        operators = {
            "$eq": lambda: value in metadata_value if isinstance(metadata_value, list) else metadata_value == value,
            "$ne": lambda: metadata_value != value,
            "$gt": lambda: metadata_value > value,
            "$gte": lambda: metadata_value >= value,
            "$lt": lambda: metadata_value < value,
            "$lte": lambda: metadata_value <= value,
            "$in": lambda: any(item in metadata_value for item in value) if isinstance(metadata_value, list) else metadata_value in value,
            "$nin": lambda: all(item not in value for item in metadata_value) if isinstance(metadata_value, list) else metadata_value not in value,
            "$and": lambda: all(apply_filter(sub_filter) for sub_filter in value),
            "$or": lambda: any(apply_filter(sub_filter) for sub_filter in value)
        }
        if operator not in operators:
            raise ValueError(f"Unsupported operator: {operator}")
        return operators[operator]()

    def apply_filter(filter_rule):
        if isinstance(filter_rule, dict):
            if all(isinstance(value, dict) for value in filter_rule.values()):
                return all(
                    apply_operator(op_key, filter_key, filter_rule[filter_key][op_key])
                    for filter_key in filter_rule
                    for op_key in filter_rule[filter_key]
                )
            elif "$and" in filter_rule or "$or" in filter_rule:
                operator = "$and" if "$and" in filter_rule else "$or"
                return apply_operator(operator, None, filter_rule[operator])
            else:
                return all(apply_operator("$eq", key, value) for key, value in filter_rule.items())
        else:
            raise ValueError(f"Invalid filter rule: {filter_rule}")

    return apply_filter(filter_dict)
"""
