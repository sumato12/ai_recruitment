from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    title: str
    description: str
    skills: str
    experience: int = Field(ge=0)


class TopicQuestionPlan(BaseModel):
    top_n: int = Field(default=5, ge=1, le=50)
    skills: int = Field(default=3, ge=0, le=20)
    dsa: int = Field(default=2, ge=0, le=20)
    oop: int = Field(default=1, ge=0, le=20)
    system_design: int = Field(default=1, ge=0, le=20)
    projects: int = Field(default=2, ge=0, le=20)

    def counts(self) -> dict[str, int]:
        return {
            "skills": self.skills,
            "dsa": self.dsa,
            "oop": self.oop,
            "system_design": self.system_design,
            "projects": self.projects,
        }

