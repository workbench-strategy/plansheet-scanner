# 🚀 AI Tools Quick Reference Card

## 🎯 **Daily Use Guidelines**

### **Cursor AI - Primary Development**
```
✅ DO:
- Use for code generation and refactoring
- Include type hints and docstrings
- Add proper error handling and logging
- Write unit tests for new functionality
- Follow existing project patterns
- Validate engineering domain accuracy

❌ DON'T:
- Accept code without error handling
- Skip type hints or documentation
- Ignore existing project structure
- Forget to test new functionality
- Use incorrect engineering terminology
```

### **GitHub Copilot - Code Completion**
```
✅ DO:
- Use for code completion and documentation
- Review all suggestions before accepting
- Ensure suggestions follow project standards
- Verify type hints and docstrings included
- Check error handling is appropriate

❌ DON'T:
- Accept suggestions without review
- Ignore project coding standards
- Skip validation of suggestions
- Forget to test completed code
```

### **Microsoft 365 Copilot - Documentation**
```
✅ DO:
- Use for technical documentation
- Include code examples when helpful
- Reference existing project docs
- Use professional, technical language
- Include relevant context and links

❌ DON'T:
- Use informal language in technical docs
- Skip code examples for complex functions
- Forget to reference existing documentation
- Ignore version control for doc changes
```

---

## 🔧 **Essential Commands**

### **Before Using AI Tools**
```bash
# Check existing codebase for similar functionality
find src/ -name "*.py" -exec grep -l "keyword" {} \;

# Review project structure
tree src/core/ -L 2

# Check coding standards
black --check src/
flake8 src/
mypy src/
```

### **After Using AI Tools**
```bash
# Run tests
pytest tests/ -v

# Check code coverage
pytest --cov=src --cov-report=term-missing

# Format code
black src/
isort src/

# Validate changes
pre-commit run --all-files
```

---

## 📋 **Quality Checklist**

### **Code Quality**
- [ ] Type hints included
- [ ] Docstrings present
- [ ] Error handling implemented
- [ ] Logging added
- [ ] Tests written
- [ ] PEP 8 compliant
- [ ] Follows project patterns

### **ML Code Quality**
- [ ] Data validation included
- [ ] Model persistence implemented
- [ ] Feature importance analysis
- [ ] Performance metrics added
- [ ] Cross-validation used
- [ ] Engineering standards validated

### **Documentation Quality**
- [ ] Clear and concise
- [ ] Code examples included
- [ ] References to existing docs
- [ ] Professional language
- [ ] Consistent formatting
- [ ] Version controlled

---

## 🚨 **Red Flags - Stop and Review**

### **Code Issues**
- No error handling
- Missing type hints
- No docstrings
- Inconsistent naming
- No tests
- Breaks existing functionality

### **ML Issues**
- No data validation
- Missing model persistence
- No performance metrics
- Incorrect engineering terminology
- No compliance validation
- Poor data quality handling

### **Documentation Issues**
- Informal language
- Missing code examples
- No references to existing docs
- Inconsistent formatting
- Outdated information

---

## 🎯 **Best Prompts**

### **Cursor AI Prompts**
```
"Create a function to [specific task] following our project standards with type hints, docstrings, error handling, and unit tests"

"Refactor this code to improve [specific aspect] while maintaining compatibility with existing systems"

"Add comprehensive logging and monitoring to this ML pipeline"
```

### **Copilot Prompts**
```
"Complete this function with proper error handling and type hints"

"Generate docstring for this function following our project standards"

"Add unit tests for this new functionality"
```

### **M365 Copilot Prompts**
```
"Create technical documentation for this feature including code examples and references to existing docs"

"Generate a project status report with performance metrics and recommendations"

"Create a troubleshooting guide for common issues with this system"
```

---

## 📞 **When to Ask for Help**

### **Technical Issues**
- Complex integration problems
- Performance optimization needs
- Security concerns
- Architecture decisions
- Domain-specific questions

### **Quality Issues**
- Code review disagreements
- Testing strategy questions
- Documentation standards
- Compliance requirements
- Best practice decisions

---

## 🔄 **Continuous Learning**

### **Weekly Review**
- Document successful AI tool usage
- Identify improvement opportunities
- Share best practices with team
- Update rules based on lessons learned

### **Monthly Assessment**
- Review AI tool effectiveness
- Update guidelines and rules
- Collect team feedback
- Plan improvements

This quick reference ensures consistent, high-quality AI tool usage while maintaining project standards! 🚀
