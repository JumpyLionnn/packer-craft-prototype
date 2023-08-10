use std::{str::Chars, iter::{Enumerate, Peekable}, fmt::Display, fs, io::Write, collections::HashMap};
use std::fmt::Write as w;
use dotenvy::dotenv;

fn main() {
    dotenv().expect(".env file not found");

    loop {
        let mut src = String::new();
        print!(">>> ");
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut src).unwrap();
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
    
        let mut output = String::new();
        let mut emitter = Emitter::new(&mut output);
        emitter.emit(node);
    
        let output_path = std::env::var("DP_PATH").expect("Variable doesnt exist");
    
        println!("displaying output for: {src}");
        println!("{output}");
        fs::write(output_path, output).unwrap();
    }
}


#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Whitespace,

    Identifier,
    Integer,

    VarKeyword,

    Plus,
    Minus,
    Star,
    Slash,
    Equal,

    OpenParenthesis,
    CloseParenthesis,

    Semicolon,
    EndOfFile,
}

#[derive(Debug, Clone)]
enum Value {
    Int(i32),
    Identifier(String)
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
                    '=' => Token::from_kind(TokenKind::Equal),
                    '(' => Token::from_kind(TokenKind::OpenParenthesis),
                    ')' => Token::from_kind(TokenKind::CloseParenthesis),
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
            _other => None
        }
    }
}

#[derive(Debug)]
struct VariableDeclaration {
    identifier: String,
    expression: Expression
}

#[derive(Debug)]
enum Statement {
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
struct AssignmentExpression {
    identifier: String,
    expression: Box<Expression>
}


#[derive(Debug)]
enum Expression {
    Binary(BinaryExpression),
    Assignment(AssignmentExpression),
    Name(String),
    Number(i32)
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
            self.next_token();
            Token::from_kind(kind)
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
            TokenKind::VarKeyword => Statement::VariableDeclaration(self.parse_variable_declaration()),
            _other => Statement::Expression({
                let exression = self.parse_expression();
                self.expect_token(TokenKind::Semicolon);
                exression
            })
        }
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
        let mut left = left.unwrap_or_else(|| self.parse_primary_expression());
        let mut precedence = self.get_binary_operator_precedence(&self.current.kind);
        while precedence > parent_precedence && precedence != 0 {
            let operator_token = self.next_token();
            let right = self.parse_primary_expression();
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
            TokenKind::Identifier => {
                let identifier = self.expect_token(TokenKind::Identifier);
                Expression::Name(identifier.value.unwrap_or(Value::Identifier(String::from(""))).get_identifier_or_default().to_string())
            },
            _other => {
                panic!("unexpected token {:?}", _other);
            }
        }
        
    }

    fn get_binary_operator_precedence(&self, kind: &TokenKind) -> u32 {
        match kind {
            TokenKind::Star |
            TokenKind::Slash => 2,
            TokenKind::Plus |
            TokenKind::Minus => 1,
            _other => 0
        }
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
    output: &'a mut String,
    scoreboard: String,
    symbols: HashMap<String, Identifier>
}

impl<'a> Emitter<'a> {
    pub fn new(output: &'a mut String) -> Self {
        Self {
            identifier: IdentifierGenerator::new("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".chars().collect()),
            output,
            scoreboard: String::from("temp"),
            symbols: HashMap::new()
        }
    }

    pub fn emit(&mut self, statements: Vec<Statement>) {
        writeln!(*self.output, "scoreboard objectives add {name} dummy \"{name}\"", name = self.scoreboard).unwrap();
        let mut identifier = None;
        for statement in statements {
            identifier = Some(self.emit_statement(statement));
        }
        if let Some(identifier) = identifier {
            writeln!(*self.output, "tellraw @a [\"\",{{\"text\":\"The result is: \"}},{{\"score\":{{\"name\":\"{}\",\"objective\":\"{}\"}}}}]", identifier, self.scoreboard).unwrap();
        }
    }

    fn emit_statement(&mut self, node: Statement) -> Identifier {
        match node {
            Statement::VariableDeclaration(declaration) => self.emit_variable_declaration(declaration),
            Statement::Expression(expression) => self.emit_expression(expression),
        }
    }

    fn emit_variable_declaration(&mut self, declaration: VariableDeclaration) -> Identifier {
        let identifier = self.emit_expression(declaration.expression);
        println!("var identifier {identifier}");
        self.symbols.insert(declaration.identifier, identifier.clone());
        identifier
    }
    
    fn emit_expression(&mut self, node: Expression) -> Identifier {
        match node {
            Expression::Assignment(assignment) => self.emit_assignment_expression(assignment),
            Expression::Binary(binary) => self.emit_binary_expression(binary),
            Expression::Number(value) => self.emit_int(value),
            Expression::Name(name) => self.emit_name(name),
        }
    }

    fn emit_assignment_expression(&mut self, assignment: AssignmentExpression) -> Identifier {
        let expression = self.emit_expression(*assignment.expression);
        let var_identifier = self.symbols.get(&assignment.identifier).unwrap();
        writeln!(self.output, "scoreboard players operation {} {name} = {} {name}", var_identifier, expression, name=self.scoreboard).unwrap();
        expression
    }

    fn emit_binary_expression(&mut self, expression: BinaryExpression) -> Identifier {
        let left = self.emit_expression(*expression.left);
        let right = self.emit_expression(*expression.right);

        let operation = match expression.operator.kind {
            TokenKind::Plus => "+=",
            TokenKind::Minus => "-=",
            TokenKind::Star => "*=",
            TokenKind::Slash => "/=",
            _other => panic!()
        };
        writeln!(*self.output, "scoreboard players operation {} {name} {} {} {name}", left, operation, right, name = self.scoreboard).unwrap();
        left
    }

    fn emit_int(&mut self, value: i32) -> Identifier {
        let identifier = self.identifier.next();
        writeln!(*self.output, "scoreboard players set {} {} {}", identifier, self.scoreboard, value).unwrap();
        identifier
    }

    fn emit_name(&mut self, name: String) -> Identifier{
        let var_identifier = self.symbols.get(&name).unwrap();
        let identifier = self.identifier.next();
        writeln!(*self.output, "scoreboard players operation {} {name} = {} {name}", identifier, var_identifier, name =  self.scoreboard).unwrap();
        identifier
    }
}