
Shader "Skybox/Escalated"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            uniform sampler2D _MainTex; 

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f {
                float4 uv : TEXCOORD0;         //posWorld
                float4 vertex : SV_POSITION;   //pos
            };

            v2f vert (appdata v) {
                appdata v2;
                v2.vertex = v.vertex; //mul(v.vertex ,float4x4(-1,0.,0.,0.,  0.,1.,0.,0.,  0.,0.,1.,0.,  0.,0.,0.,1.));
                v2f o = (v2f)0;
                o.uv = mul(unity_ObjectToWorld, v2.vertex);
                o.vertex = UnityObjectToClipPos( v.vertex); // * float4(1.0,1.,1.0,1.) ); // squish skybox sphere
                return o;
            }

float2x2 m=float2x2(.8,-.6,.6,.8);


float g(float2 p){
    float e=abs(sin(p.x+sin(p.y)));
    p=mul(m,p);
    return .1*(e+sin(p.x+sin(p.y)));
}

float n(float2 p){
    p*=.1;
    float s=5.,t=.9;
    for(int i=0;i<9;i++)
        t-=s*g(p),s*=.4,p=mul(m,2.1*p)+t;
    return 3.-exp(t);
}

//void mainImage(out vec4 fragColor, in vec2 fragCoord){

         fixed4 frag (v2f v2) : SV_Target
            {
                float2 fragCoord = v2.vertex;

                float3 viewDirection = normalize(v2.uv.xyz- _WorldSpaceCameraPos.xyz  );
                fixed4 fragColor = tex2D(_MainTex, v2.uv);
                
                float3 rd = viewDirection;                                                        // ray direction for fragCoord.xy
                float3 ro = _WorldSpaceCameraPos.xyz*.0001;                                             // ray origin

    float v=_Time.y*2.,u=sin(v*.1),x=.0,p=.0,o=.0;
    float3 r=float3(fragCoord.xy-1.,0),z,y;  //float3(fragCoord/iResolution.xy-1.,0),z,y;
    for(int d=0;d<288;d++)        
        if (p*.0002<=x)
            z=float3(0,-8.*g(float2(0,v)*.1),v)+p*normalize(float3(r.x-u,r.y*.3+.1,2)),x=z.y+n(z.xz),p+=x,o++;
    x=n(z.xz);
    y=normalize(float3(n(z.xz-float2(.01,0))-x,0,n(z.xz-float2(0,.01))-x-n(z.zx*11.)*.002));
    fragColor.xyz=dot(float3(-.5,-.5,-.5),y)*n(z.zx*6.)*float3(.1,.2,.3)+.1+o*.002+log(p)*.1;
                return fragColor;
            }

            ENDCG
        }
    }
}



