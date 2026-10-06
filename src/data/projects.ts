export const projects = [
  { slug: 'qwen3-runtime', number: '01', field: '推理系统', title: 'Qwen3 Runtime', description: '面向多轮 Agent 的单 GPU 推理运行时。从请求调度、分页 KV 到会话暂停与恢复，沿完整执行路径分析性能。', topics: ['请求调度', 'KV 状态管理', 'GPU 执行优化'], repo: 'https://github.com/yuzheng310/qwen3-runtime' },
  { slug: 'repocompass', number: '02', field: 'Agent 后训练', title: 'RepoCompass', description: '仓库级代码定位 Agent 后训练系统。基于 CodeScout，围绕文件与函数定位构建任务环境、可验证奖励与训练流程。', topics: ['任务环境与奖励', 'RFT → GSPO', '训练工程'], repo: null },
  { slug: 'database', number: '03', field: '数据库系统', title: 'DynamoDB 兼容与在线迁移', description: '实习期间参与的团队项目。负责兼容层与配套迁移工具的架构和研发，处理 API 语义、增量同步与一致性校验。', topics: ['API 语义兼容', '在线迁移', '一致性校验'], repo: null },
];
