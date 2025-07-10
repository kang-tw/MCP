# MCP (Model Context Protocol) - 취약점 데이터셋 컬렉션

본 리포지토리는 패치 자동화 연구를 위한 보안 취약점 코드 데이터셋을 수집·정리한 것입니다.

## 🎯 목적
- 패치 자동화 기술 개발을 위한 학습 데이터 제공
- 다양한 취약점 유형별 코드 예제 수집
- 보안 연구 및 교육용 자료 구축

## 📊 데이터셋 카테고리

### 1. 종합 취약점 데이터셋
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [security-patches-dataset](https://github.com/security-commits/security-patches-dataset) | CVE, NVD, OSV 등 다양한 소스에서 취합한 취약점 예측 데이터셋 | CVE 기반 다양한 취약점 | 대규모 취약점 패치 데이터로 자동화 모델 학습에 적합 |
| [software-vulnerability-datasets](https://github.com/vulnerability-dataset/software-vulnerability-datasets) | CVE Details, 정적 분석 도구, 버전 관리 메타데이터 포함 | 소프트웨어 취약점 전반 | 정적 분석 결과와 패치 정보가 함께 제공되어 패치 자동화 연구에 유용 |
| [SecurityEval](https://github.com/s2e-lab/SecurityEval) | ML 기반 코드 생성 기술 평가용 취약점 예제 | 코드 생성 시 발생하는 취약점 | AI 기반 패치 생성 모델의 성능 평가 기준으로 활용 가능 |

### 2. 프로그래밍 언어별 취약점 예제

#### C/C++ 취약점
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [Damn_Vulnerable_C_Program](https://github.com/hardik05/Damn_Vulnerable_C_Program) | C 언어 일반적인 취약점 예제 | Buffer Overflow, Memory Leak 등 | 저수준 언어 취약점 패치 자동화 연구용 |

#### JavaScript 취약점
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [javascript-security-vulnerabilities](https://github.com/kushal-kamra/javascript-security-vulnerabilities) | JavaScript 보안 취약점 코드 예제 | XSS, Injection 공격 등 | 웹 애플리케이션 취약점 패치 자동화 학습용 |
| [writing-secure-javascript](https://github.com/Blaumaus/writing-secure-javascript) | JavaScript 보안 취약점과 예방 기법 | 일반적인 JS 취약점 | 취약한 코드와 보안 코드 비교 학습용 |

#### Python 취약점
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [Web-Security-Vulnerabilities](https://github.com/verylazytech/Web-Security-Vulnerabilities) | Python 웹 보안 취약점 예제와 수정 버전 | 웹 애플리케이션 취약점 | 취약점과 수정 코드 쌍으로 제공되어 패치 학습에 최적 |
| [python-scripts](https://github.com/testcomputer/python-scripts) | Python 보안 도구 및 취약점 스캐닝 예제 | 다양한 보안 취약점 | 보안 도구 개발과 취약점 탐지 자동화에 활용 |

#### .NET 취약점
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [StaticSecurityCodeScan.Net6.Examples](https://github.com/SumitMSunhala/StaticSecurityCodeScan.Net6.Examples) | .NET 보안 취약점과 SecurityCodeScan 도구 활용 | .NET 특화 취약점 | 엔터프라이즈 애플리케이션 취약점 패치 자동화용 |

### 3. 웹 애플리케이션 취약점

#### OWASP 기반 취약점
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [OWASP Juice Shop](https://github.com/juice-shop/juice-shop) | 현대적이고 정교한 취약한 웹 애플리케이션 | OWASP Top 10 전체 | 실제 웹 애플리케이션 수준의 복잡한 취약점 패치 연구용 |
| [OWASP WebGoat PHP](https://github.com/OWASP/OWASPWebGoatPHP) | 의도적으로 취약하게 만든 웹 애플리케이션 | 웹 보안 학습용 취약점 | 체계적인 웹 취약점 학습 및 패치 자동화 훈련용 |
| [Mutillidae](https://github.com/webpwnized/mutillidae) | 웹 보안 훈련용 취약한 웹 애플리케이션 | 다양한 웹 취약점 | 실습 환경에서 패치 자동화 도구 테스트용 |

#### API 보안 취약점
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [vulnerable-rest-api](https://github.com/bnematzadeh/vulnerable-rest-api) | OWASP API Top 10 기반 취약한 REST API | API 보안 취약점 | 현대 마이크로서비스 아키텍처의 API 보안 패치 자동화용 |
| [capital](https://github.com/Checkmarx/capital) | OWASP API Top 10 기반 취약한 API 애플리케이션 | API 취약점 전반 | API 보안 CTF 환경에서 패치 기술 검증용 |

### 4. 특수 도메인 취약점

#### 블록체인/스마트 컨트랙트
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [smart-contract-security](https://github.com/badgerblockchain/smart-contract-security) | 스마트 컨트랙트 취약점 예제와 예방법 | 블록체인 특화 취약점 | 블록체인 코드 보안 패치 자동화 연구용 |
| [SWCVulnerableCode](https://github.com/SaurabhSinghDev/SWCVulnerableCode) | SWC-100~120 Solidity 취약점 예제 | Solidity 스마트 컨트랙트 취약점 | 스마트 컨트랙트 취약점 분류 및 패치 패턴 학습용 |

#### 모바일 애플리케이션
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [iGoat-Swift](https://github.com/OWASP/iGoat-Swift) | iOS Swift 취약한 애플리케이션 | 모바일 보안 취약점 | 모바일 앱 보안 패치 자동화 연구용 |
| [DVNA](https://github.com/appsecco/dvna) | Node.js 기반 취약한 애플리케이션 | Node.js 웹 취약점 | 서버사이드 JavaScript 취약점 패치 학습용 |

### 5. 종합 보안 가이드

#### 보안 코딩 가이드
| 리포지토리 | 설명 | 취약점 유형 | 수집 이유 |
|-----------|------|------------|-----------|
| [Secure-Code-Snippets-for-Each-Vulnerability](https://github.com/ferid333/Secure-Code-Snippets-for-Each-Vulnerability) | 다양한 취약점에 대한 안전한 코드 예제 | 범용 취약점 유형 | 취약점별 보안 코딩 패턴 학습 및 패치 생성 참조용 |
| [VulnerabilityInsights](https://github.com/MYathin21/VulnerabilityInsights) | 프로그래밍 언어별 취약점 구조화된 컬렉션 | 다양한 언어의 취약점 | 언어별 취약점 특성 이해 및 패치 자동화 모델 훈련용 |
| [secure-coding-examples](https://github.com/sahildari/secure-coding-examples) | 보안 코딩 실무 예제 모음 | 실무적 보안 취약점 | 실제 개발 환경에서의 패치 자동화 적용 연구용 |

## 🔍 주요 취약점 유형별 분류

### 1. 인젝션 공격 (Injection)
- SQL Injection, XSS, Command Injection
- **관련 리포지토리**: sql-injection-prevention-guide, xss-example

### 2. 인증 및 세션 관리 결함
- 약한 인증, 세션 하이재킹
- **관련 리포지토리**: OWASP Juice Shop, Mutillidae

### 3. 민감 데이터 노출
- 암호화 누락, 키 관리 오류
- **관련 리포지토리**: javascript-security-share-my-place

### 4. 보안 설정 오류
- 기본 설정 사용, 불필요한 기능 활성화
- **관련 리포지토리**: OWASP Top 10 Examples

### 5. 버퍼 오버플로우
- 메모리 관리 오류, 스택/힙 오버플로우
- **관련 리포지토리**: Damn_Vulnerable_C_Program

### 6. API 보안 결함
- 부적절한 인증, 과도한 데이터 노출
- **관련 리포지토리**: vulnerable-rest-api, capital

## 📈 데이터셋 활용 방안

### 1. 학습 데이터 구축
- 취약한 코드 → 패치된 코드 쌍 생성
- 취약점 유형별 패턴 분석
- 다양한 프로그래밍 언어별 특성 학습

### 2. 모델 훈련
- 코드 변환 모델 (Vulnerable → Secure)
- 취약점 탐지 모델
- 패치 품질 평가 모델

### 3. 평가 벤치마크
- 패치 자동화 도구 성능 평가
- 다양한 취약점 유형에 대한 커버리지 측정
- 실제 환경에서의 적용 가능성 검증

## 🚀 향후 계획

1. **데이터셋 확장**: 더 많은 언어와 프레임워크의 취약점 예제 수집
2. **품질 개선**: 중복 제거, 라벨링 정확도 향상
3. **자동화 도구 개발**: 새로운 취약점 데이터 자동 수집 시스템
4. **평가 메트릭 개발**: 패치 품질 평가를 위한 표준 메트릭 정의

## 📝 라이센스 및 사용 주의사항

- 모든 데이터는 **연구 및 교육 목적**으로만 사용
- 각 리포지토리의 개별 라이센스 준수 필요
- 악의적 목적으로의 사용 금지
- 실제 시스템에 적용 전 충분한 테스트 필요

## 🤝 기여 방법

1. 새로운 취약점 데이터셋 발견 시 Issue 생성
2. 데이터 품질 개선 제안
3. 분류 체계 개선 아이디어 제공
4. 자동화 도구 개발 참여

---

**⚠️ 중요**: 본 리포지토리의 모든 취약점 코드는 교육 및 연구 목적으로만 제공됩니다. 실제 운영 환경에서는 절대 사용하지 마세요.
