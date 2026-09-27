"""Execute the three saved-results notebooks in process (no socket service)."""
import os
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[1]
for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
 notebook=nbformat.read(path,as_version=4)
 os.chdir(path.parent)
 InteractiveShell.clear_instance();shell=InteractiveShell.instance();number=0
 for cell in notebook.cells:
  if cell.cell_type!='code':continue
  number+=1
  with capture_output() as captured:result=shell.run_cell(cell.source,store_history=True)
  if not result.success:raise RuntimeError(f'{path.name}: {result.error_in_exec or result.error_before_exec}')
  cell.execution_count=number;cell.outputs=[]
  if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
  if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
  for output in captured.outputs:cell.outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
 notebook.metadata['execution_method']='Executed against saved result snapshots; full fitting is invoked by reproduce.py.'
 nbformat.validate(notebook);nbformat.write(notebook,path)
 print(f'Executed {path.name}: {number} code cells')
