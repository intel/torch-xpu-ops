import ast

class DebugPrinter(ast.NodeTransformer):
    def __init__(self, function):
        self.import_added = False
        self.function = function
        self.scope_stack = []
    
    def visit_Module(self, node):
        if not self.import_added:
            node.body = [
                ast.Import(names=[ast.alias(name='torch')]),
                ast.Import(names=[ast.alias(name='numpy', asname='np')])
            ] + node.body
            self.import_added = True
        node.body = [self.visit(n) for n in node.body]
        return node
    
    def visit_FunctionDef(self, node):
        self.scope_stack.append(node.name)
        if node.name == self.function:
            debug_print = self._create_print(
                    f"FUNC {node.name}({', '.join(arg.arg for arg in node.args.args)})"
                )

            node.body = [debug_print] + [self.visit(n) for n in node.body]
        self.scope_stack.pop()
        return node
    
    def visit_Expr(self, node):
        return self._process_statement(node)
    
    def visit_Assign(self, node):
        return self._process_statement(node)
    
    def _process_statement(self, node):
        debug_nodes = []
        
        if len(self.scope_stack) != 0 and self.scope_stack[-1] == self.function:
            if hasattr(node, 'lineno'):
                stmt = ast.unparse(node).strip().replace('\n', ' ')[:80]
                debug_nodes.append(self._create_print(f"LINE {node.lineno}: {stmt}"))
            
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                debug_nodes.extend(self._debug_call(node.value))
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                debug_nodes.extend(self._debug_call(node.value))
            
            for name in self._get_names(node):
                debug_nodes.append(self._create_var_debug(name))
        
        return debug_nodes + [node]
    
    def _debug_call(self, call_node):
        debug_nodes = []
        func_name = ast.unparse(call_node.func)
        debug_nodes.append(self._create_print(f"CALL {func_name}"))
        
        for i, arg in enumerate(call_node.args):
            if isinstance(arg, ast.Name):
                debug_nodes.append(self._create_var_debug(arg.id, f"ARG {i+1}"))
        
        for kw in call_node.keywords:
            if isinstance(kw.value, ast.Name):
                debug_nodes.append(self._create_var_debug(kw.value.id, f"KWARG {kw.arg}"))
        
        return debug_nodes
    
    def _get_names(self, node):
        names = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                names.add(n.id)
        return names
    
    def _create_var_debug(self, var_name, prefix=None):
        """Safe variable debugging that handles all types"""
        debug_code = f"""
try:
    if isinstance({var_name}, (torch.Tensor, np.ndarray)):
        print(f"{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: {{type({var_name}).__name__}} = {{repr({var_name})[:250]}} shape={{tuple({var_name}.shape)}} dtype={{str({var_name}.dtype)}}")
    elif hasattr({var_name}, 'shape') and hasattr({var_name}, 'dtype'):
        print(f"{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: {{type({var_name}).__name__}} = {{repr({var_name})[:250]}} shape={{tuple({var_name}.shape)}} dtype={{str({var_name}.dtype)}}")
    else:
        print(f"{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: {{type({var_name}).__name__}} = {{repr({var_name})[:250]}}")
except Exception as e:
    print(f"{'VAR ' + prefix + ': ' if prefix else 'VAR '}{var_name}: [Error during inspection: {{str(e)}}]")
"""
        return ast.parse(debug_code).body[0]
    
    def _create_print(self, message):
        return ast.Expr(value=ast.Call(
            func=ast.Name(id='print', ctx=ast.Load()),
            args=[ast.Constant(value=message)],
            keywords=[ast.keyword(arg='flush', value=ast.Constant(value=True))]
        ))

def debug_transform(source_code, function):
    tree = ast.parse(source_code)
    transformer = DebugPrinter(function)
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)

try:
    with open('test/test_foreach.py', 'r') as file:
        content = file.read()
except FileNotFoundError:
    print("Error: The file 'my_file.txt' was not found.")

source = content

function = "test_parity"
print(debug_transform(source, function))
