# 🤖 AI Tools Rules & Guidelines

## 🎯 Overview

This document provides comprehensive rules and guidelines for using AI tools (Copilot, Cursor, Microsoft 365) in the plansheet scanner project. These rules ensure consistent, high-quality AI assistance while maintaining code quality and project standards.

---

## 🖥️ **Cursor AI Rules**

### **Core Development Rules**

#### **Code Generation & Refactoring**
```
You are an expert Python developer working on a machine learning-powered plansheet scanner for engineering drawings. Follow these rules:

**Project Context:**
- This is a production ML system for analyzing engineering plans and traffic drawings
- Code must be production-ready with proper error handling and logging
- All code must follow PEP 8 and include comprehensive type hints
- Use the existing project structure in src/core/ for new functionality

**Code Quality Standards:**
- Always include docstrings for all functions and classes
- Use type hints for all parameters and return values
- Implement proper error handling with specific exception types
- Add logging with appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Write unit tests for all new functionality using pytest
- Follow the existing naming conventions in the codebase

**ML/Data Science Standards:**
- Validate all input data before processing
- Handle missing values appropriately
- Use scikit-learn for traditional ML models
- Use PyTorch for deep learning models
- Save models using joblib for scikit-learn or torch.save for PyTorch
- Include feature importance analysis for tree-based models

**Engineering Domain Knowledge:**
- Understand traffic engineering and signal design
- Know civil engineering drawing standards
- Be familiar with MUTCD and traffic control devices
- Understand as-built drawing analysis and validation
- Know engineering review processes and compliance requirements

**When generating code:**
1. Check existing codebase for similar functionality first
2. Follow established patterns and conventions
3. Ensure compatibility with existing systems
4. Add appropriate logging and monitoring
5. Include comprehensive error handling
6. Write corresponding unit tests
7. Update documentation if needed
```

#### **File Organization Rules**
```
**File Structure Guidelines:**
- Keep related functionality in the same module
- Use __init__.py files to define public APIs
- Group imports: standard library, third-party, local
- Maintain consistent file structure across modules
- Place new functionality in appropriate src/core/ subdirectories

**Naming Conventions:**
- Use descriptive variable and function names
- Follow snake_case for functions and variables
- Use PascalCase for classes
- Use UPPER_CASE for constants
- Prefix private methods with underscore
```

#### **Testing Rules**
```
**Testing Requirements:**
- Write unit tests for all new functionality
- Test files should be in tests/ directory
- Use descriptive test names that explain expected behavior
- Include both positive and negative test cases
- Mock external dependencies appropriately
- Achieve minimum 80% code coverage
- Use pytest fixtures for common test setup
- Test edge cases and error conditions
```

### **ML-Specific Rules**

#### **Model Development**
```
**ML Model Guidelines:**
- Always validate input data types and shapes
- Handle missing or invalid data gracefully
- Provide feature importance analysis for interpretable models
- Include model persistence and loading functionality
- Add performance metrics and evaluation
- Consider model versioning and compatibility
- Use cross-validation for model evaluation
- Implement early stopping for deep learning models
- Save model checkpoints during training
```

#### **Data Processing**
```
**Data Processing Standards:**
- Validate data quality before processing
- Handle outliers and missing values appropriately
- Use vectorized operations when possible
- Implement data versioning and tracking
- Add data validation schemas
- Log data processing steps and statistics
- Implement data pipeline monitoring
```

---

## 🤖 **GitHub Copilot Rules**

### **Code Completion Rules**

#### **General Guidelines**
```
**Copilot Configuration:**
- Use Copilot for code completion, not full file generation
- Always review and validate Copilot suggestions
- Ensure suggestions follow project coding standards
- Verify type hints and docstrings are included
- Check that error handling is appropriate
- Validate that suggestions use existing project patterns

**Acceptance Criteria:**
- Code must include proper type hints
- Functions must have comprehensive docstrings
- Error handling must be specific and informative
- Code must follow PEP 8 formatting
- Variable names must be descriptive and clear
- Logic must be correct and efficient
```

#### **ML Code Completion**
```
**ML-Specific Guidelines:**
- Ensure data validation is included
- Verify model hyperparameters are reasonable
- Check that feature engineering follows best practices
- Validate that model evaluation metrics are appropriate
- Ensure model persistence is implemented correctly
- Verify that preprocessing steps are consistent
```

### **Documentation Rules**
```
**Documentation Standards:**
- Generate comprehensive docstrings for all functions
- Include parameter descriptions and types
- Document return values and exceptions
- Add usage examples for complex functions
- Include references to relevant documentation
- Maintain consistent documentation style
```

---

## 📊 **Microsoft 365 Copilot Rules**

### **Documentation & Communication**

#### **Technical Documentation**
```
**Documentation Guidelines:**
- Use clear, concise language for technical concepts
- Include code examples when explaining functionality
- Reference existing project documentation
- Maintain consistent formatting and structure
- Include diagrams or screenshots when helpful
- Update documentation when code changes
- Use version control for documentation changes
```

#### **Project Communication**
```
**Communication Standards:**
- Use professional, technical language
- Reference specific code files and line numbers
- Include error messages and stack traces when relevant
- Provide context for technical decisions
- Use consistent terminology across documents
- Include links to relevant resources and documentation
```

### **Data Analysis & Reporting**

#### **ML Results Analysis**
```
**Analysis Guidelines:**
- Present results in clear, actionable format
- Include confidence intervals and uncertainty measures
- Provide context for engineering decisions
- Use appropriate visualizations for data types
- Include statistical significance where applicable
- Reference relevant engineering standards
- Document assumptions and limitations
```

#### **Performance Reporting**
```
**Reporting Standards:**
- Include baseline comparisons
- Show improvement metrics over time
- Provide context for performance changes
- Include cost-benefit analysis where relevant
- Document model accuracy and reliability
- Include recommendations for improvements
```

---

## 🔧 **Integration Rules**

### **Cross-Tool Workflow**

#### **Development Workflow**
```
**Tool Integration Guidelines:**
1. Use Cursor for primary code development and refactoring
2. Use Copilot for code completion and documentation
3. Use M365 Copilot for documentation and communication
4. Ensure consistency across all tools
5. Validate output from each tool before proceeding
6. Maintain version control for all changes
7. Update documentation when making changes
```

#### **Quality Assurance**
```
**Quality Standards:**
- Review all AI-generated code before committing
- Run tests to validate functionality
- Check code coverage requirements
- Verify documentation accuracy
- Ensure compliance with project standards
- Validate engineering domain accuracy
- Test integration with existing systems
```

### **Error Handling & Validation**

#### **AI Tool Validation**
```
**Validation Process:**
1. Review AI suggestions for technical accuracy
2. Verify domain knowledge is correct
3. Check that suggestions follow project patterns
4. Validate that error handling is appropriate
5. Ensure logging and monitoring are included
6. Test functionality before accepting suggestions
7. Update tests when adding new features
```

---

## 📋 **Best Practices Checklist**

### **Before Using AI Tools**
- [ ] Understand the specific task requirements
- [ ] Review existing codebase for similar functionality
- [ ] Identify appropriate tool for the task
- [ ] Prepare clear, specific prompts
- [ ] Have validation criteria ready

### **During AI Tool Usage**
- [ ] Review suggestions carefully before accepting
- [ ] Validate technical accuracy and domain knowledge
- [ ] Ensure code follows project standards
- [ ] Check for proper error handling and logging
- [ ] Verify type hints and documentation

### **After AI Tool Usage**
- [ ] Run tests to validate functionality
- [ ] Check code coverage requirements
- [ ] Update documentation if needed
- [ ] Commit changes with descriptive messages
- [ ] Review integration with existing systems

---

## 🚨 **Common Pitfalls to Avoid**

### **Code Generation Issues**
- Don't accept code without proper error handling
- Don't skip type hints and documentation
- Don't ignore existing project patterns
- Don't accept suggestions without testing
- Don't forget to update tests for new functionality

### **Domain Knowledge Issues**
- Don't accept incorrect engineering terminology
- Don't ignore relevant standards and codes
- Don't skip validation against engineering requirements
- Don't forget to consider real-world constraints
- Don't ignore compliance and safety requirements

### **Integration Issues**
- Don't break existing functionality
- Don't ignore backward compatibility
- Don't skip integration testing
- Don't forget to update documentation
- Don't ignore performance implications

---

## 📚 **Reference Resources**

### **Project Documentation**
- `README.md` - Project overview and setup
- `agent_prompts.md` - Existing agent prompts
- `pyproject.toml` - Project configuration
- `tests/` - Test examples and patterns

### **External Resources**
- PEP 8 - Python style guide
- scikit-learn documentation
- PyTorch documentation
- MUTCD standards
- Traffic engineering references

### **Tool-Specific Resources**
- Cursor documentation and best practices
- GitHub Copilot guidelines
- Microsoft 365 Copilot documentation
- Project-specific coding standards

---

## 🔄 **Continuous Improvement**

### **Regular Reviews**
- Monthly review of AI tool effectiveness
- Quarterly updates to rules and guidelines
- Annual assessment of tool integration
- Continuous feedback collection from team
- Regular updates to domain knowledge base

### **Feedback Loop**
- Document successful AI tool usage patterns
- Identify areas for improvement
- Share best practices across team
- Update rules based on lessons learned
- Maintain knowledge base of effective prompts

This comprehensive set of rules ensures that AI tools enhance productivity while maintaining the high quality and engineering accuracy required for the plansheet scanner project! 🚀
