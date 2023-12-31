use std::{str::Chars, iter::{Enumerate, Peekable}, fmt::Display, fs, collections::HashMap};
use std::fmt::Write as w;
use dotenvy::dotenv;

fn main() {
    dotenv().expect(".env file not found");

    //loop {
        // let mut src = String::new();
        // print!(">>> ");
        // std::io::stdout().flush().unwrap();
        // std::io::stdin().read_line(&mut src).unwrap();
        let src = fs::read_to_string("example.mc").unwrap();
        let mut tokenizer = Tokenizer::new(&src);
    
        let mut tokens = Vec::new();
        let mut token = tokenizer.next_token();
        while token.kind != TokenKind::EndOfFile {
            tokens.push(token);
            token = tokenizer.next_token();
        }
        
        let mut parser = Parser::new(&tokens);
        let node = parser.parse();

        println!("{:#?}", node);
    
        let mut output = HashMap::new();
        let mut emitter = Emitter::new(&mut output, "testdp");
        emitter.emit(node, String::from("load"), String::from("tick"));

    
        let output_path = std::env::var("DP_PATH").expect("Variable doesnt exist");
    
        println!("displaying output for: {src}");
        for (name, code) in output.iter() {
            println!("{name}:\n{code}");
            let mut path = String::new();
            write!(&mut path, "{output_path}{name}.mcfunction").unwrap();
            fs::write(path, code).unwrap();
        }
    }
//}


#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Whitespace,

    Identifier,
    Integer,
    String,

    VarKeyword,
    IfKeyword,
    ElseKeyword,
    TrueKeyword,
    FalseKeyword,

    Plus,
    Minus,
    Star,
    Slash,
    Equal,

    EqualEqual,
    GreatedThan,
    LessThan,
    ExclamationMark,

    OpenParenthesis,
    CloseParenthesis,
    OpenCurlyBrace,
    CloseCurlyBrace,

    Comma,
    Semicolon,
    EndOfFile,
}

#[derive(Debug, Clone)]
enum Value {
    Int(i32),
    Identifier(String),
    String(String)
}

impl Value {
    pub fn get_int_or_default(&self) -> i32 {
        if let Self::Int(value) = self {
            *value
        }
        else {
            0
        }
    }

    pub fn get_identifier_or_default(&self) -> &str {
        if let Self::Identifier(value) = self {
            value
        }
        else {
            ""
        }
    }

    pub fn get_string_or_default(&self) -> &str {
        if let Self::String(value) = self {
            value
        }
        else {
            ""
        }
    }
}

#[derive(Debug, Clone)]
struct Token {
    kind: TokenKind,
    value: Option<Value>
}

impl Token {
    fn from_kind(kind: TokenKind) -> Self {
        Self {
            kind,
            value: None
        }
    }

    fn from_number(number: i32) -> Self {
        Self {
            kind: TokenKind::Integer,
            value: Some(Value::Int(number))
        }
    }

    fn from_identifier(identifier: &str) -> Self {
        Self {
            kind: TokenKind::Identifier,
            value: Some(Value::Identifier(identifier.to_string()))
        }
    }

    fn from_string(value: String) -> Self {
        Self {
            kind: TokenKind::String,
            value: Some(Value::String(value))
        }
    }
}

struct Tokenizer<'a> {
    it: Peekable<Enumerate<Chars<'a>>>,
    text: &'a String
}

impl<'a> Tokenizer<'a> {
    pub fn new(text: &'a String) -> Self {
        Self { it: text.chars().enumerate().peekable(), text: text}
    }

    fn next_token(&mut self) -> Token  {

        let c = self.it.next();
        match c {
            Some((mut index, c)) => {
                match c {
                    '+' => Token::from_kind(TokenKind::Plus),
                    '-' => Token::from_kind(TokenKind::Minus),
                    '*' => Token::from_kind(TokenKind::Star),
                    '/' => Token::from_kind(TokenKind::Slash),
                    '=' => {
                        if self.it.peek().is_some_and(|(_index, c)| *c == '=') {
                            self.it.next();
                            Token::from_kind(TokenKind::EqualEqual)
                        }
                        else {
                            Token::from_kind(TokenKind::Equal)
                        }
                    },
                    '>' => Token::from_kind(TokenKind::GreatedThan),
                    '<' => Token::from_kind(TokenKind::LessThan),
                    '!' => Token::from_kind(TokenKind::ExclamationMark),
                    '(' => Token::from_kind(TokenKind::OpenParenthesis),
                    ')' => Token::from_kind(TokenKind::CloseParenthesis),
                    '{' => Token::from_kind(TokenKind::OpenCurlyBrace),
                    '}' => Token::from_kind(TokenKind::CloseCurlyBrace),
                    ',' => Token::from_kind(TokenKind::Comma),
                    ';' => Token::from_kind(TokenKind::Semicolon),
                    '0'..='9' => {
                        let start_index = index;
                        while self.it.peek().is_some_and(|(_index, c)| c.is_digit(10)) {
                            (index, _) = self.it.next().unwrap();
                        }
                        let end_index = index;
                        let num_str = &self.text[start_index..=end_index];
                        let number = num_str.parse::<i32>().unwrap();
                        Token::from_number(number)
                    },
                    'a'..='z' | 'A'..='Z' => {
                        let start_index = index;
                        while self.it.peek().is_some_and(|(_index, c)| c.is_ascii_alphabetic()) {
                            (index, _) = self.it.next().unwrap();
                        }
                        let end_index = index;
                        let identifier = &self.text[start_index..=end_index];
                        if let Some(keyword) = self.get_keyword(identifier) {
                            Token::from_kind(keyword)
                        }
                        else {
                            Token::from_identifier(identifier)
                        }
                    },
                    '"' => {
                        let start_index = index;
                        while self.it.peek().is_some_and(|(_index, c)| *c != '"') {
                            (index, _) = self.it.next().unwrap();
                        }
                        self.it.next();
                        let end_index = index;
                        let value = &self.text[start_index + 1..=end_index];
                        Token::from_string(value.to_string())
                    },
                    _ if c.is_whitespace() => {
                        while self.it.peek().is_some_and(|(_index, c)| c.is_whitespace()) {
                            self.it.next().unwrap();
                        }
                        Token::from_kind(TokenKind::Whitespace)
                    }
                    _other => {
                        // bad character
                        self.next_token()
                    }
                }
            },
            None => Token::from_kind(TokenKind::EndOfFile),
        }
    }

    fn get_keyword(&self, identifier: &str) -> Option<TokenKind> {
        match identifier {
            "var" => Some(TokenKind::VarKeyword),
            "if" => Some(TokenKind::IfKeyword),
            "else" => Some(TokenKind::ElseKeyword),
            "true" => Some(TokenKind::TrueKeyword),
            "false" => Some(TokenKind::FalseKeyword),
            _other => None
        }
    }
}


#[derive(Debug)]
struct IfStatement {
    condition: Expression,
    statement: Box<Statement>,
    else_clause: Option<Box<Statement>>
}

#[derive(Debug)]
struct VariableDeclaration {
    identifier: String,
    expression: Expression
}

#[derive(Debug)]
enum Statement {
    Block(Vec<Statement>),
    If(IfStatement),
    VariableDeclaration(VariableDeclaration),
    Expression(Expression)
}


#[derive(Debug)]
struct BinaryExpression {
    left: Box<Expression>,
    operator: Token,
    right: Box<Expression>
}

#[derive(Debug)]
struct UnaryExpression {
    operator: Token,
    operand: Box<Expression>
}


#[derive(Debug)]
struct AssignmentExpression {
    identifier: String,
    expression: Box<Expression>
}

#[derive(Debug)]
struct CallExpression {
    func_name: String,
    arguments: Vec<Expression>
}

#[derive(Debug)]
enum Expression {
    Binary(BinaryExpression),
    Unary(UnaryExpression),
    Assignment(AssignmentExpression),
    Call(CallExpression),
    Name(String),
    Number(i32),
    Boolean(bool),
    String(String)
}

struct Parser<'a> {
    current: Token,
    it: Peekable<std::slice::Iter<'a, Token>>
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a Vec<Token>) -> Self {
        let mut it = tokens.iter().peekable();
        Self {
            current: it.next().unwrap().clone(),
            it
        }
    }

    fn next_token(&mut self) -> Token {
        let previous = self.current.clone();
        self.current = self.it.next().unwrap_or(&Token::from_kind(TokenKind::EndOfFile)).clone();
        while self.current.kind == TokenKind::Whitespace {
            self.current = self.it.next().unwrap_or(&Token::from_kind(TokenKind::EndOfFile)).clone();
        }
        previous
    }

    fn expect_token(&mut self, kind: TokenKind) -> Token {
        if self.current.kind == kind {
            self.next_token()
        }
        else {
            panic!("expected {:?} found token {:?}", kind, self.current.kind);
            // self.next_token();
            // Token::from_kind(kind)
        }
    }

    fn peek(&self, offset: usize) -> Token {
        let mut it = self.it.clone();
        let mut current_offset = 0;
        let mut next = it.next();
        while next.is_some_and(|token| token.kind == TokenKind::Whitespace) || current_offset < offset - 1 {
            next = it.next();
            if next.is_some_and(|token| token.kind != TokenKind::Whitespace) || next.is_none() {
                current_offset += 1;
            }
        }
        next.unwrap_or(&Token::from_kind(TokenKind::EndOfFile)).clone()
    }

    pub fn parse(&mut self) -> Vec<Statement> {
        println!("{:?} {:?}", self.current, self.peek(2));
        let mut statements = Vec::new();
        while self.current.kind != TokenKind::EndOfFile {
            statements.push(self.parse_statement());
        }
        statements
    }

    fn parse_statement(&mut self) -> Statement {
        match &self.current.kind {
            TokenKind::OpenCurlyBrace => Statement::Block(self.parse_block_statement()),
            TokenKind::IfKeyword => Statement::If(self.parse_if_statement()),
            TokenKind::VarKeyword => Statement::VariableDeclaration(self.parse_variable_declaration()),
            _other => Statement::Expression({
                let exression = self.parse_expression();
                self.expect_token(TokenKind::Semicolon);
                exression
            })
        }
    }

    fn parse_block_statement(&mut self) -> Vec<Statement> {
        let mut statements = Vec::new();
        self.expect_token(TokenKind::OpenCurlyBrace);
        while self.current.kind != TokenKind::CloseCurlyBrace && self.current.kind != TokenKind::EndOfFile {
            statements.push(self.parse_statement());
        }
        self.expect_token(TokenKind::CloseCurlyBrace);
        statements
    }

    fn parse_if_statement(&mut  self) -> IfStatement {
        self.expect_token(TokenKind::IfKeyword);
        let condition = self.parse_expression();
        let statement = self.parse_statement();
        let mut else_clause = None;
        if self.current.kind == TokenKind::ElseKeyword {
            self.next_token();
            else_clause = Some(Box::new(self.parse_statement()));
        }
        IfStatement { condition, statement: Box::new(statement), else_clause }
    }

    fn parse_variable_declaration(&mut self) -> VariableDeclaration {
        self.expect_token(TokenKind::VarKeyword);
        let identifier = self.expect_token(TokenKind::Identifier).value.unwrap_or(Value::Identifier(String::from(""))).get_identifier_or_default().to_string();
        self.expect_token(TokenKind::Equal);
        let expression = self.parse_expression();
        self.expect_token(TokenKind::Semicolon);
        VariableDeclaration { identifier, expression }
    }

    fn parse_expression(&mut self) -> Expression {
        if self.current.kind == TokenKind::Identifier && self.peek(1).kind == TokenKind::Equal {
            Expression::Assignment(self.parse_assignment_expression())
        }
        else {
            self.parse_binary_expression()
        }
    }

    fn parse_assignment_expression(&mut self) -> AssignmentExpression {
        let identifier = self.expect_token(TokenKind::Identifier).value.unwrap_or(Value::Identifier(String::from(""))).get_identifier_or_default().to_string();
        self.expect_token(TokenKind::Equal);
        let expression = self.parse_expression();
        AssignmentExpression { 
            identifier, 
            expression: Box::new(expression)
        }
    }

    fn parse_binary_expression(&mut self) -> Expression {
        self.parse_binary_expression_inner(0, None)
    }

    fn parse_binary_expression_inner(&mut self, parent_precedence: u32, left: Option<Expression>) -> Expression {
        let mut left = left.unwrap_or_else(|| self.parse_unary_or_primary_expression());
        let mut precedence = self.get_binary_operator_precedence(&self.current.kind);
        while precedence > parent_precedence && precedence != 0 {
            let operator_token = self.next_token();
            let right = self.parse_unary_or_primary_expression();
            let next_precedence = self.get_binary_operator_precedence(&self.current.kind);
            if next_precedence > precedence {
                let right = self.parse_binary_expression_inner(precedence, Some(right));
                left = Expression::Binary(BinaryExpression{ left: Box::new(left), operator: operator_token, right: Box::new(right) });
            }
            else if next_precedence < precedence && parent_precedence > 0 {
                return Expression::Binary(BinaryExpression { left: Box::new(left), operator: operator_token, right: Box::new(right) });
            }
            else {
                left = Expression::Binary(BinaryExpression { left: Box::new(left), operator: operator_token, right: Box::new(right) });
            }

            precedence = self.get_binary_operator_precedence(&self.current.kind);
            
        }
        left
    }

    fn parse_unary_or_primary_expression(&mut self) -> Expression {
        if self.is_unary_operator(&self.current.kind) {
            let operator = self.next_token();
            let expression = self.parse_unary_or_primary_expression();
            Expression::Unary(UnaryExpression { 
                operator, 
                operand: Box::new(expression) 
            })
        }
        else {
            self.parse_primary_expression()
        }
    }

    fn parse_primary_expression(&mut self) -> Expression {
        match &self.current.kind {
            TokenKind::OpenParenthesis => {
                self.next_token();
                let expression = self.parse_expression();
                self.expect_token(TokenKind::CloseParenthesis);
                expression
            },
            TokenKind::Integer => {
                let number_token = self.expect_token(TokenKind::Integer);
                Expression::Number(number_token.value.unwrap_or(Value::Int(0)).get_int_or_default())
            },
            TokenKind::TrueKeyword => {
                Expression::Boolean(true)
            },
            TokenKind::FalseKeyword => {
                Expression::Boolean(false)
            },
            TokenKind::String => {
                let string_token = self.expect_token(TokenKind::String);
                Expression::String(string_token.value.unwrap_or(Value::Int(0)).get_string_or_default().to_owned())
            },
            TokenKind::Identifier => self.parse_name_or_call_expression(),
            _other => {
                panic!("unexpected token {:?}", _other);
            }
        }
        
    }

    fn parse_name_or_call_expression(&mut self) -> Expression {
        if self.peek(1).kind == TokenKind::OpenParenthesis {
            self.parse_call_expression()
        }
        else {
            self.parse_name_expression()
        }
    }

    fn parse_call_expression(&mut self) -> Expression {
        let identifier = self.expect_token(TokenKind::Identifier);
        self.expect_token(TokenKind::OpenParenthesis);
        let mut arguments = Vec::new();
        while self.current.kind != TokenKind::CloseParenthesis && self.current.kind != TokenKind::EndOfFile {
            arguments.push(self.parse_expression());
            println!("argument {:?}", arguments.last());
            if self.current.kind != TokenKind::CloseParenthesis && self.current.kind != TokenKind::EndOfFile {
                self.expect_token(TokenKind::Comma);
            }
        }
        self.expect_token(TokenKind::CloseParenthesis);
        Expression::Call(CallExpression { 
            func_name: identifier.value.unwrap_or(Value::Identifier(String::from(""))).get_identifier_or_default().to_string(), 
            arguments 
        })
    }

    fn parse_name_expression(&mut self) -> Expression {
        let identifier = self.expect_token(TokenKind::Identifier);
        Expression::Name(identifier.value.unwrap_or(Value::Identifier(String::from(""))).get_identifier_or_default().to_string())
    }

    fn get_binary_operator_precedence(&self, kind: &TokenKind) -> u32 {
        match kind {
            TokenKind::Star |
            TokenKind::Slash => 2,
            TokenKind::Plus |
            TokenKind::Minus => 1,
            TokenKind::GreatedThan |
            TokenKind::LessThan => 1,
            TokenKind::EqualEqual => 1,
            _other => 0
        }
    }

    fn is_unary_operator(&self, kind: &TokenKind) -> bool {
        *kind == TokenKind::Minus || *kind == TokenKind::Plus || *kind == TokenKind::ExclamationMark
    }
}


#[derive(Clone)]
struct Identifier {
    name: String
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

struct IdentifierGenerator {
    chars_indices: Vec<usize>,
    avaliable_characters: Vec<char>
}

impl IdentifierGenerator {
    pub fn new(characters: Vec<char>) -> Self {
        Self {
            chars_indices: vec![0],
            avaliable_characters: characters
        }
    }

    pub fn next(&mut self) -> Identifier {
        let mut name = String::with_capacity(self.chars_indices.len());
        for index in self.chars_indices.iter().rev() {
            name.push(self.avaliable_characters[*index]);
        }
        self.increment();
        Identifier { name }
    }

    fn increment(&mut self) {
        match self.chars_indices.iter_mut().find(|index| **index < self.avaliable_characters.len() - 1) {
            Some(index) => *index += 1,
            None => self.chars_indices.push(0),
        }
    }
}

struct Emitter<'a> {
    identifier: IdentifierGenerator,
    func_identifier: IdentifierGenerator,
    output: &'a mut HashMap<String, String>,
    scoreboard: String,
    symbols_stack: Vec<HashMap<String, Identifier>>,
    function_stack: Vec<String>,
    datapack_namespace: &'a str
}

impl<'a> Emitter<'a> {
    pub fn new(output: &'a mut HashMap<String, String>, namespace: &'a str) -> Self {
        Self {
            identifier: IdentifierGenerator::new("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect()),
            func_identifier: IdentifierGenerator::new("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect()),
            output,
            scoreboard: String::from("temp"),
            symbols_stack: vec![HashMap::new()],
            function_stack: Vec::new(),
            datapack_namespace: namespace
        }
    }

    fn start_function(&mut self) {
        self.function_stack.push(String::new());
    }

    fn end_function(&mut self) -> Identifier {
        let identifier = self.func_identifier.next();
        self.output.insert(identifier.clone().name, self.function_stack.pop().unwrap());
        identifier
    }

    pub fn emit(&mut self, statements: Vec<Statement>, load_name: String, tick_name: String) {
        self.function_stack.push(String::new());
        writeln!(self.function_stack.last_mut().unwrap(), "scoreboard objectives add {name} dummy \"{name}\"", name = self.scoreboard).unwrap();
        self.output.insert(load_name, self.function_stack.pop().unwrap());

        self.function_stack.push(String::new());
        for statement in statements {
            self.emit_statement(statement);
        }
        self.output.insert(tick_name, self.function_stack.pop().unwrap());
    }

    fn emit_statement(&mut self, node: Statement) -> Option<Identifier> {
        match node {
            Statement::Block(block) => self.emit_block_statement(block),
            Statement::If(statement) => self.emit_if_statement(statement),
            Statement::VariableDeclaration(declaration) => Some(self.emit_variable_declaration(declaration)),
            Statement::Expression(expression) => Some(self.emit_expression(expression)),
        }
    }
    
    fn emit_block_statement(&mut self, block: Vec<Statement>) -> Option<Identifier> {
        self.symbols_stack.push(HashMap::new());
        let mut identifier = None;
        for statement in block {
            identifier = self.emit_statement(statement);
        }
        self.symbols_stack.pop();
        identifier
    }

    fn emit_if_statement(&mut self, statement: IfStatement) -> Option<Identifier> {
        let condition = self.emit_expression(statement.condition);

        self.start_function();
        self.emit_statement(*statement.statement);
        let func = self.end_function();
        writeln!(self.function_stack.last_mut().unwrap(), "execute if score {} {} matches 1.. run function {}:{}", condition, self.scoreboard, self.datapack_namespace, func).unwrap();
        
        if let Some(else_clause) = statement.else_clause {
            self.start_function();
            self.emit_statement(*else_clause);
            let else_func = self.end_function();
            writeln!(self.function_stack.last_mut().unwrap(), "execute unless score {} {} matches 1.. run function {}:{}", condition, self.scoreboard, self.datapack_namespace, else_func).unwrap();
        }
        None
    }

    fn emit_variable_declaration(&mut self, declaration: VariableDeclaration) -> Identifier {
        let identifier = self.emit_expression(declaration.expression);
        println!("var identifier {identifier}");
        self.symbols_stack.last_mut().unwrap().insert(declaration.identifier, identifier.clone());
        identifier
    }
    
    fn emit_expression(&mut self, node: Expression) -> Identifier {
        match node {
            Expression::Assignment(assignment) => self.emit_assignment_expression(assignment),
            Expression::Binary(binary) => self.emit_binary_expression(binary),
            Expression::Unary(unary) => self.emit_unary_expression(unary),
            Expression::Number(value) => self.emit_int(value),
            Expression::Call(call) => self.emit_call_expression(call),
            Expression::Name(name) => self.emit_name(name),
            Expression::Boolean(value) => {
                let value = if value {1} else {0};
                self.emit_int(value)
            },
            Expression::String(_) => unimplemented!(),
        }
    }

    fn emit_assignment_expression(&mut self, assignment: AssignmentExpression) -> Identifier {
        let expression = self.emit_expression(*assignment.expression);
        let var_identifier = self.lookup_symbol(&assignment.identifier).unwrap();
        writeln!(self.function_stack.last_mut().unwrap(), "scoreboard players operation {} {name} = {} {name}", var_identifier, expression, name=self.scoreboard).unwrap();
        expression
    }

    fn emit_binary_expression(&mut self, expression: BinaryExpression) -> Identifier {
        let left = self.emit_expression(*expression.left);
        let right = self.emit_expression(*expression.right);

        if is_arithmitic_operator(&expression.operator.kind) {
            let operation = match expression.operator.kind {
                TokenKind::Plus => "+=",
                TokenKind::Minus => "-=",
                TokenKind::Star => "*=",
                TokenKind::Slash => "/=",
                _other => panic!()
            };
            writeln!(self.function_stack.last_mut().unwrap(), "scoreboard players operation {} {name} {} {} {name}", left, operation, right, name = self.scoreboard).unwrap();
            left
        }
        else if is_relational_operator(&expression.operator.kind) {
            let operation = match expression.operator.kind {
                TokenKind::EqualEqual => "=",
                TokenKind::GreatedThan => ">",
                TokenKind::LessThan => "<",
                _other => panic!()
            };
            let identifier = self.identifier.next();
            writeln!(self.function_stack.last_mut().unwrap(), "execute if score {left} {name} {} {} {name} run scoreboard players set {identifier} {name} 1 ", operation, right, name = self.scoreboard).unwrap();
            writeln!(self.function_stack.last_mut().unwrap(), "execute unless score {left} {name} {} {} {name} run scoreboard players set {identifier} {name} 0 ", operation, right, name = self.scoreboard).unwrap();
            identifier
        }
        else {
            panic!("unknown operator");
        }
    }

    fn emit_unary_expression(&mut self, unary: UnaryExpression) -> Identifier {
        if unary.operator.kind == TokenKind::Minus {
            let expression = self.emit_expression(*unary.operand);
            let result = self.identifier.next();
            writeln!(self.function_stack.last_mut().unwrap(), "scoreboard players reset {} {}", result, self.scoreboard).unwrap();
            writeln!(self.function_stack.last_mut().unwrap(), "scoreboard players operation {} {name} -= {} {name}", result, expression, name=self.scoreboard).unwrap();
            result
        }
        else if unary.operator.kind == TokenKind::ExclamationMark {
            let expression = self.emit_expression(*unary.operand);
            let result = self.identifier.next();
            writeln!(self.function_stack.last_mut().unwrap(), "execute if score {} {name} matches 1.. run scoreboard players set {} {name} 0", expression, result, name=self.scoreboard).unwrap();
            writeln!(self.function_stack.last_mut().unwrap(), "execute if score {} {name} matches 0 run scoreboard players set {} {name} 1", expression, result, name=self.scoreboard).unwrap();
            result
        }
        else {
            self.emit_expression(*unary.operand)
        }
    }

    fn emit_int(&mut self, value: i32) -> Identifier {
        let identifier = self.identifier.next();
        writeln!(self.function_stack.last_mut().unwrap(), "scoreboard players set {} {} {}", identifier, self.scoreboard, value).unwrap();
        identifier
    }

    fn emit_call_expression(&mut self, call: CallExpression) -> Identifier {
        // let arguments = call.arguments.into_iter().map(|expression| {
        //     self.emit_expression(expression)
        // }).collect::<Vec<Identifier>>();

        match &call.func_name[..] {
            "log" => {
                assert!(call.arguments.len() > 0);
                let mut command = String::new();
                write!(command, "tellraw @a [\"\",{{\"text\":\"[{}][log] \"}}", self.datapack_namespace).unwrap();
                call.arguments.into_iter().for_each(|expression| {
                    if let Expression::String(text) = expression {
                        write!(command, ",{{\"text\":\"{}\"}}", text).unwrap();
                    }
                    else {
                        let argument = self.emit_expression(expression);
                        write!(command, ",{{\"score\":{{\"name\":\"{}\",\"objective\":\"{}\"}}}}", argument, self.scoreboard).unwrap();
                    }
                });
                // for argument in arguments {
                //     write!(self.function_stack.last_mut().unwrap(), ",{{\"score\":{{\"name\":\"{}\",\"objective\":\"{}\"}}}}, {{\"text\":\" \"}}", argument, self.scoreboard).unwrap();
                // }
                write!(command, "]").unwrap();
                writeln!(self.function_stack.last_mut().unwrap(), "{}", command).unwrap();
                self.emit_int(0)
            },
            _other => {
                panic!();
            }
        }
    }

    fn emit_name(&mut self, name: String) -> Identifier{
        let var_identifier = self.lookup_symbol(&name).unwrap();
        let identifier = self.identifier.next();
        writeln!(self.function_stack.last_mut().unwrap(), "scoreboard players operation {} {name} = {} {name}", identifier, var_identifier, name =  self.scoreboard).unwrap();
        identifier
    }

    fn lookup_symbol(&self, name: &String) -> Option<Identifier> {
        let mut identifier = None;
        for frame in self.symbols_stack.iter().rev() {
            identifier = frame.get(name);
            if identifier.is_some() {
                break;
            }
        }
        identifier.cloned()
    }

}

fn is_arithmitic_operator(kind: &TokenKind) -> bool {
    *kind == TokenKind::Plus || *kind == TokenKind::Minus || *kind == TokenKind::Star || *kind == TokenKind::Slash
}

fn is_relational_operator(kind: &TokenKind) -> bool {
    *kind == TokenKind::EqualEqual || *kind == TokenKind::GreatedThan || *kind == TokenKind::LessThan
}