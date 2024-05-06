from typing import Union, List, Dict

from llama_index.core import PromptTemplate

FIGURE_TABLE_EXT_PROMPT_TMPL = PromptTemplate(
    """Given the figure from a research paper, please extract only the variables and observations names of the figure/chart as columns header and rows index in an HTML table, but do not extract any numerical data values.
Figure information is below.
---------------------
{header_str}
--------------------- 
Answer:""")


def stringify_to_instructions(obj: Union[List, Dict], conjunction='or') -> str:
    if isinstance(obj, dict):
        items = list(obj)
    elif isinstance(obj, list):
        items = obj
    else:
        return obj.__repr__()

    if len(items) > 2:
        repr_str = ', '.join(str(item) for item in items[:-1]) + f', {conjunction} ' + str(items[-1])
    else:
        repr_str = ', '.join(str(item) for item in items)

    return repr_str
