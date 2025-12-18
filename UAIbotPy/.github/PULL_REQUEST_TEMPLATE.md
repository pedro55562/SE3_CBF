### Description

Clearly describe the purpose of this PR. If addressing an issue, reference it like:  
`Fixes #123` or `Related to #456`

### Type of Change

- [ ] Documentation Update
- [ ] Bug Fix
- [ ] New Feature
- [ ] Refactoring
- [ ] Breaking Change*

\* Requires justification below

### Checklists

#### Documentation Changes

- [ ] Updated all affected documentation files
- [ ] Verified documentation builds correctly
- [ ] Ensured consistency with existing style
- [ ] Added relevant examples if applicable

#### Code Changes

- [ ] Added/updated docstrings
- [ ] All existing tests pass
- [ ] New tests cover changes
- [ ] Type hints maintained/updated
- [ ] No unintended side effects

#### Build & Compatibility

- [ ] Validates with current noxtests.yaml matrix:
  - Python versions: 3.10 → 3.13
  - Platforms: Ubuntu, Windows, macOS
- [ ] If breaking builds, provide:
    ```markdown
    ### Breaking Change Justification
    **Affected Components:**  
    [Which parts stop working]
    
    **Technical Necessity:**  
    [Why changes are unavoidable]
    
    **Migration Path:**  
    [How users should adapt]
    ```